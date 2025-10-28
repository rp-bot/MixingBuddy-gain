import yaml
import os
from collections import defaultdict, Counter
from pathlib import Path

def extract_all_instruments(metadata_dir):
    """Extract all unique instruments from MedleyDB metadata files"""
    
    all_instruments = set()
    instrument_counts = Counter()
    track_instruments = {}
    
    metadata_path = Path(metadata_dir)
    
    for yaml_file in metadata_path.glob("*_METADATA.yaml"):
        try:
            with open(yaml_file, 'r') as f:
                data = yaml.safe_load(f)
            
            track_id = yaml_file.stem.replace("_METADATA", "")
            track_instruments[track_id] = []
            
            # Extract instruments from stems
            if 'stems' in data:
                for stem_id, stem_data in data['stems'].items():
                    instrument = stem_data.get('instrument', '')
                    
                    # Handle both single instruments and lists
                    if isinstance(instrument, list):
                        for inst in instrument:
                            if inst and inst.strip():
                                all_instruments.add(inst.strip())
                                instrument_counts[inst.strip()] += 1
                                track_instruments[track_id].append(inst.strip())
                    elif instrument and instrument.strip():
                        all_instruments.add(instrument.strip())
                        instrument_counts[instrument.strip()] += 1
                        track_instruments[track_id].append(instrument.strip())
                    
                    # Also check raw tracks for more detailed instrument info
                    if 'raw' in stem_data:
                        for raw_id, raw_data in stem_data['raw'].items():
                            raw_instrument = raw_data.get('instrument', '')
                            if isinstance(raw_instrument, list):
                                for inst in raw_instrument:
                                    if inst and inst.strip():
                                        all_instruments.add(inst.strip())
                                        instrument_counts[inst.strip()] += 1
                            elif raw_instrument and raw_instrument.strip():
                                all_instruments.add(raw_instrument.strip())
                                instrument_counts[raw_instrument.strip()] += 1
                                
        except Exception as e:
            print(f"Error processing {yaml_file}: {e}")
            continue
    
    return sorted(all_instruments), instrument_counts, track_instruments

def list_all_tracks(metadata_dir):
    """List all tracks in the MedleyDB metadata directory"""
    metadata_path = Path(metadata_dir)
    all_tracks = []
    for i, yaml_file in enumerate(metadata_path.glob("*_METADATA.yaml")):
        # a yaml file name is like this: AClassicEducation_NightOwl_METADATA.yaml
        # so we need to get the track name from the yaml file name
        track_name = yaml_file.stem.replace("_METADATA", "")
        print(f"{i+1:3d}. {track_name}")
        all_tracks.append(track_name)
    return all_tracks

# Usage
metadata_dir = "/home/rpbot/Documents/GitHub/research/automatic-mixing/milestone_0/data/medleydb_package/medleydb/data/Metadata"
medleydb_dir = "/home/rpbot/Documents/GitHub/research/automatic-mixing/milestone_0/data/medleydb"

all_instruments, instrument_counts, track_instruments = extract_all_instruments(metadata_dir)

# lets list all the instruments in the metadata directory
print("\n\nAll instruments in MedleyDB:\n")
for i, instrument in enumerate(all_instruments, 1):
    print(f"{i:3d}. {instrument}")
    # lets count the number of tracks that have this instrument
    count = instrument_counts[instrument]
    print(f"  - {instrument} (appears in {count} tracks)")
