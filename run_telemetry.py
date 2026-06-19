import hardware_manager as h


target_backend = "ibm_fez" 

print(f"Fetching calibration data for {target_backend}...")


h.dump_backend_telemetry(target_backend, outfile="hardware_telemetry.json")

print("Telemetry update complete.")