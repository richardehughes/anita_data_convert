import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
import ROOT

parser = argparse.ArgumentParser()
parser.add_argument("--ANITA_ROOT_DATA", type=str, required=True)
parser.add_argument("--run", type=int, required=True)
parser.add_argument(
    "--num_parts",
    type=int,
    default=1,
    help="Total number of chunks to split the run into",
)
parser.add_argument(
    "--part_index",
    type=int,
    default=0,
    help="Index of the current chunk (0 to num_parts - 1)",
)
args = parser.parse_args()

RUN_NUMBER = args.run
DATA_DIR = "/fs/ess/PAS2159/"
os.environ["ANITA_ROOT_DATA"] = args.ANITA_ROOT_DATA

# Create part-specific filename if running partitioned
if args.num_parts > 1:
  OUTPUT_FILE = f"/fs/scratch/PAS3311/anita_data/anita_data_parquet/run{RUN_NUMBER}_part{args.part_index:02d}of{args.num_parts:02d}_calibrated_maps.parquet"
else:
  OUTPUT_FILE = f"/fs/scratch/PAS3311/anita_data/anita_data_parquet/run{RUN_NUMBER}_calibrated_maps.parquet"
# if args.num_parts > 1:
#   OUTPUT_FILE = f"/fs/ess/PAS3311/anita/anita_data_parquet/run{RUN_NUMBER}_part{args.part_index:02d}of{args.num_parts:02d}_calibrated_maps.parquet"
# else:
#   OUTPUT_FILE = f"/fs/ess/PAS3311/anita/anita_data_parquet/run{RUN_NUMBER}_calibrated_maps.parquet"


def set_anita_version(version=3):
  ROOT.gInterpreter.Declare('#include "AnitaVersion.h"')
  ROOT.gInterpreter.Declare('#include "FFTtools.h"')
  ROOT.AnitaVersion.set(version)


def GetArray(graph):
  phiNum = graph.GetNbinsX()
  thetaNum = graph.GetNbinsY()
  arr = np.zeros((phiNum, thetaNum), dtype=np.float32)
  for theta_bin in range(thetaNum):
    for phi_bin in range(phiNum):
      arr[phi_bin, theta_bin] = graph.GetBinContent(phi_bin + 1, theta_bin + 1)
  return np.rot90(arr)


def get_waveform_data(graph):
  N = graph.GetN()
  times = np.zeros(N, dtype=np.float64)
  volts = np.zeros(N, dtype=np.float64)
  for k in range(N):
    times[k] = graph.GetX()[k]
    volts[k] = graph.GetY()[k]
  return volts, times


def calculate_energy_selected_snr(x_img, rms_v, rms_h):
  """Calculates SNR based on the antenna channel with the highest total energy.

  Parameters:
      x_img: np.ndarray of shape (N_events, 48, time_bins, 2)
             axis 3 corresponds to polarization (0: H-Pol, 1: V-Pol)
      rms_v: float, RMS noise for Vertical polarization
      rms_h: float, RMS noise for Horizontal polarization

  Returns:
      tuple (snr_h, snr_v): 1D np.ndarrays containing SNR per event.
  """
  n_events = x_img.shape[0]
  results = []
  const_rms = [rms_h, rms_v]

  for pol_idx in [0, 1]:
    waves = x_img[:, :, :, pol_idx]
    energy = np.sum(waves**2, axis=2)
    best_ant_indices = np.argmax(energy, axis=1)

    rows = np.arange(n_events)
    best_waveforms = waves[rows, best_ant_indices]
    v_peak = np.max(np.abs(best_waveforms), axis=1)

    snr = v_peak / (const_rms[pol_idx] + 1e-9)
    results.append(snr)

  return results[0], results[1]


def main():
  set_anita_version(3)

  print(f"Opening Run {RUN_NUMBER} from {DATA_DIR}...")
  d = ROOT.AnitaDataset(RUN_NUMBER, False, ROOT.WaveCalType.kDefault)
  total_events = d.N()
  print(f"Total events found in dataset: {total_events}")

  # Disjoint event partitioning across chunks
  if not (0 <= args.part_index < args.num_parts):
    raise ValueError(
        f"part_index ({args.part_index}) must be between 0 and"
        f" {args.num_parts - 1}"
    )

  start_idx = (total_events * args.part_index) // args.num_parts
  end_idx = (total_events * (args.part_index + 1)) // args.num_parts
  events_in_chunk = end_idx - start_idx

  print(
      f"Processing chunk {args.part_index + 1}/{args.num_parts}: event indices"
      f" [{start_idx}:{end_idx}] ({events_in_chunk} events)"
  )

  # Setup UCorrelator Analysis Configuration
  cfg = ROOT.UCorrelator.AnalysisConfig()
  cfg.r_time_shift_correction = True
  cfg.simulation_time_shift_correction = False
  cfg.delay_to_center = True
  cfg.use_bin_center = True

  cfg.zoomed_nphi = 80
  cfg.zoomed_dphi = 0.25
  cfg.zoomed_ntheta = 80
  cfg.zoomed_dtheta = 0.25

  Analyzer = ROOT.UCorrelator.Analyzer(cfg, True)

  rings = [
      ROOT.AnitaRing.kTopRing,
      ROOT.AnitaRing.kMiddleRing,
      ROOT.AnitaRing.kBottomRing,
  ]
  interestingBins = 150
  full_waveform_bins = 260
  output = []
  all_raw_waveforms = []

  t0 = time.time()
  t00 = time.time()

  for count, idx in enumerate(range(start_idx, end_idx), start=1):
    if count % 100 == 0:
      print(
          f"Processed {count}/{events_in_chunk} events in this chunk"
          f" (global #{idx + 1}/{total_events})..."
      )
      print(f"Elapsed for last 100: {time.time() - t0:.2f}s")
      t0 = time.time()

    d.getEntry(idx)
    usefulEvent = d.useful()
    event_number = usefulEvent.eventNumber
    event_time = d.header().realTime

    filteredEvent = ROOT.FilteredAnitaEvent(
        usefulEvent,
        ROOT.UCorrelator.getStrategyWithKey("sinsub_10_3_ad_2"),
        d.gps(),
        d.header(),
    )
    eventSummary = ROOT.AnitaEventSummary()
    Analyzer.analyze(filteredEvent, eventSummary)

    # 1. Full & Zoomed Correlation Maps
    correlationMap = np.zeros(
        (2, cfg.correlator_ntheta, cfg.correlator_nphi), dtype=np.float32
    )
    correlationMap[0] = GetArray(
        Analyzer.getCorrelationMap(ROOT.AnitaPol.kHorizontal)
    )
    correlationMap[1] = GetArray(
        Analyzer.getCorrelationMap(ROOT.AnitaPol.kVertical)
    )

    zoomCorrelationMap = np.zeros(
        (2, cfg.zoomed_ntheta, cfg.zoomed_nphi), dtype=np.float32
    )
    zoomCorrelationMap[0] = GetArray(
        Analyzer.getZoomedCorrelationMap(ROOT.AnitaPol.kHorizontal, 0)
    )
    zoomCorrelationMap[1] = GetArray(
        Analyzer.getZoomedCorrelationMap(ROOT.AnitaPol.kVertical, 0)
    )

    # 2. Calibrated Waveforms (Raw 260-sample array for SNR + 150-sample timingMap)
    event_waveforms = np.zeros(
        (len(rings) * 16, full_waveform_bins, 2), dtype=np.float32
    )
    for ring_idx, ring in enumerate(rings):
      for phi in range(16):
        ant_idx = ring_idx * 16 + phi

        graphH = usefulEvent.getGraph(ring, phi, ROOT.AnitaPol.kHorizontal)
        voltsH, _ = get_waveform_data(graphH)
        nbins_h = min(len(voltsH), full_waveform_bins)
        event_waveforms[ant_idx, :nbins_h, 0] = voltsH[:nbins_h]

        graphV = usefulEvent.getGraph(ring, phi, ROOT.AnitaPol.kVertical)
        voltsV, _ = get_waveform_data(graphV)
        nbins_v = min(len(voltsV), full_waveform_bins)
        event_waveforms[ant_idx, :nbins_v, 1] = voltsV[:nbins_v]

    all_raw_waveforms.append(event_waveforms)
    timingMap = np.transpose(event_waveforms[:, :interestingBins, :], (2, 0, 1))

    # 3. Append to Output
    output.append({
        "run": RUN_NUMBER,
        "event": event_number,
        "realTime": event_time,
        "phi_estimated": eventSummary.peak[1][0].phi,
        "theta_estimated": -eventSummary.peak[1][0].theta,
        "dphi_rough": eventSummary.peak[1][0].dphi_rough,
        "dtheta_rough": -eventSummary.peak[1][0].dtheta_rough,
        "latitude": eventSummary.anitaLocation.latitude,
        "longitude": eventSummary.anitaLocation.longitude,
        "altitude": eventSummary.anitaLocation.altitude,
        "heading": eventSummary.anitaLocation.heading,
        "timingMap": timingMap.flatten(),
        "correlationMap": correlationMap.flatten(),
        "zoomCorrelationMap": zoomCorrelationMap.flatten(),
    })

  # Build DataFrame
  df = pd.DataFrame(output)

  # Full-chunk noise RMS and energy-selected SNR calculation
  if len(all_raw_waveforms) > 0:
    x_raw = np.stack(all_raw_waveforms, axis=0)
    noise_sample = x_raw[:, :, 180:, :]
    constant_rms_hpol = float(np.sqrt(np.mean(noise_sample[:, :, :, 0] ** 2)))
    constant_rms_vpol = float(np.sqrt(np.mean(noise_sample[:, :, :, 1] ** 2)))

    snr_calc_hpol, snr_calc_vpol = calculate_energy_selected_snr(
        x_raw, rms_v=constant_rms_vpol, rms_h=constant_rms_hpol
    )
    df["energy_selected_snr_h"] = snr_calc_hpol
    df["energy_selected_snr_v"] = snr_calc_vpol
  else:
    df["energy_selected_snr_h"] = np.array([], dtype=np.float32)
    df["energy_selected_snr_v"] = np.array([], dtype=np.float32)

  # print the snr v/h for the first 10 events
  print("\nFirst 5 Events SNR Comparison:")
  for i in range(min(5, len(df))):
    print(f"  Event {df['event'].iloc[i]} | "
          f"Calculated SNR (H/V): {df['energy_selected_snr_h'].iloc[i]:.2f} / {df['energy_selected_snr_v'].iloc[i]:.2f}")
  # Save to Parquet
  os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
  df.to_parquet(OUTPUT_FILE, index=False)

  print(
      f"\nChunk complete: {events_in_chunk} events processed in"
      f" {time.time() - t00:.2f} seconds."
  )
  print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
  main()