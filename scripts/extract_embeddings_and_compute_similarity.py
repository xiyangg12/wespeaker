import os
from pathlib import Path
import torch
import pandas as pd
from wespeaker.cli.speaker import Speaker
import librosa
import soundfile as sf
import numpy as np

MODEL_NAME_LIST = [
    "/home/xiyali/git/wespeaker/exp/cnceleb_resnet34/models",
    "/home/xiyali/git/wespeaker/exp/cnceleb_resnet34_LM/models",
    "/home/xiyali/git/wespeaker/exp/voxblink2_samresnet34/models",
    "/home/xiyali/git/wespeaker/exp/voxblink2_samresnet34_ft/models",
    "/home/xiyali/git/wespeaker/exp/voxblink2_samresnet100/models",
    "/home/xiyali/git/wespeaker/exp/voxblink2_samresnet100_ft/models",
    "/home/xiyali/git/wespeaker/exp/voxceleb_resnet34/models",
    "/home/xiyali/git/wespeaker/exp/voxceleb_resnet34_LM/models",
    "/home/xiyali/git/wespeaker/exp/voxceleb_resnet152_LM/models",
    "/home/xiyali/git/wespeaker/exp/voxceleb_resnet221_LM/models",
    "/home/xiyali/git/wespeaker/exp/voxceleb_resnet293_LM/models"
]

SPEAKER_ORDER = ["VF19B", "VF19D", "VF21A", "VF21B", "VF21C", "VF21D", "VF23B", "VF23C", "VF26A", "VF32A"]
LANGUAGE_ORDER = ["can", "eng"]

def compute_similarity_matrices(audio_folder_path: str):
    """Compute similarity matrices for all wav files in a folder using multiple models."""
    wav_file_paths_can_utt1 = [os.path.join(audio_folder_path, "can", f"{speaker}_can_utt1.wav") for speaker in SPEAKER_ORDER]
    wav_file_paths_can_utt2 = [os.path.join(audio_folder_path, "can", f"{speaker}_can_utt2.wav") for speaker in SPEAKER_ORDER]
    wav_file_paths_eng_utt1 = [os.path.join(audio_folder_path, "eng", f"{speaker}_eng_utt1.wav") for speaker in SPEAKER_ORDER]
    wav_file_paths_eng_utt2 = [os.path.join(audio_folder_path, "eng", f"{speaker}_eng_utt2.wav") for speaker in SPEAKER_ORDER]

    trial_data, trial_id = [], 1
    trial_data, trial_id = _get_df_from_two_wav_path_list(wav_file_paths_can_utt1, wav_file_paths_can_utt2, trial_data, trial_id)
    trial_data, trial_id = _get_df_from_two_wav_path_list(wav_file_paths_can_utt1, wav_file_paths_eng_utt2, trial_data, trial_id)
    trial_data, trial_id = _get_df_from_two_wav_path_list(wav_file_paths_eng_utt1, wav_file_paths_can_utt2, trial_data, trial_id)
    trial_data, trial_id = _get_df_from_two_wav_path_list(wav_file_paths_eng_utt1, wav_file_paths_eng_utt2, trial_data, trial_id)
    
    # Create DataFrame with all trials and model columns
    df_columns = ['trial_id', 'stim1', 'stim2', 'spkr1', 'spkr2', 'lang1', 'lang2', 
                  'utt1', 'utt2', 'utterance_pair', 'speaker_identity', 'stimulus_language']
    df_columns.extend(MODEL_NAME_LIST)

    for model_name in MODEL_NAME_LIST:
        print(f"\nProcessing model: {model_name}")
        model = Speaker(model_name)
        
        # Compute similarities for each trial
        for trial in trial_data:
            i = trial['row_idx']
            j = trial['col_idx']
            if "can" in trial['lang1'].lower():
                wav_file_paths_1 = wav_file_paths_can_utt1 
            else:
                wav_file_paths_1 = wav_file_paths_eng_utt1
            if "can" in trial['lang2'].lower():
                wav_file_paths_2 = wav_file_paths_can_utt2
            else:
                wav_file_paths_2 = wav_file_paths_eng_utt2
            similarity = model.compute_similarity(
                wav_file_paths_1[i], 
                wav_file_paths_2[j]
            )
            trial[model_name] = similarity
        
        print(f"Computed {len(trial_data)} similarities for model {model_name}")
    
    # Remove row_idx and col_idx from trial data before creating DataFrame
    for trial in trial_data:
        trial.pop('row_idx', None)
        trial.pop('col_idx', None)
    
    df = pd.DataFrame(trial_data)
    
    # Reorder columns to match percept_data.csv format
    df = df[df_columns]
    
    # Save as CSV
    os.makedirs("./similarity_matrices", exist_ok=True)
    csv_path = os.path.join("./similarity_matrices", "model_similarities.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved all results to {csv_path}")
    
    return df


def _get_df_from_two_wav_path_list(wav_file_paths_1, wav_file_paths_2, trial_data, trial_id: int):
    n_files = len(wav_file_paths_1)
    for i in range(n_files):
        for j in range(n_files):
            if j >= i:
                lang1 = "can" if "can" in wav_file_paths_1[i] else "eng"
                lang2 = "can" if "can" in wav_file_paths_2[j] else "eng"
                utterance_pair = f"{lang1[0]}1_{lang2[0]}2"
                spkr1 = Path(wav_file_paths_1[i]).stem.split('_')[0]
                spkr2 = Path(wav_file_paths_2[j]).stem.split('_')[0]
                
                # Determine speaker identity and stimulus language
                speaker_identity = "same" if spkr1 == spkr2 else "diff"
                stimulus_language = "mixed" if lang1 != lang2 else lang1
                
                trial_info = {
                    'trial_id': f"T{trial_id:03d}",
                    'stim1': f"{spkr1}_{lang1.capitalize()}_utt1.wav",
                    'stim2': f"{spkr2}_{lang2.capitalize()}_utt2.wav",
                    'spkr1': spkr1,
                    'spkr2': spkr2,
                    'lang1': lang1.capitalize(),
                    'lang2': lang2.capitalize(),
                    'utt1': f"utt_{lang1[0]}1",
                    'utt2': f"utt_{lang2[0]}2",
                    'utterance_pair': utterance_pair,
                    'speaker_identity': speaker_identity,
                    'stimulus_language': stimulus_language,
                    'row_idx': i,
                    'col_idx': j,
                }
                trial_data.append(trial_info)
                trial_id += 1
    
    return trial_data, trial_id

def save_embeddings(audio_folder_path: str):
    """Compute and save embeddings for all wav files in a folder using multiple models."""
    os.makedirs("embeddings", exist_ok=True)
    embedding_shape = {}
    for model_name in MODEL_NAME_LIST:
        save_embeddings_for_model(audio_folder_path, model_name, embedding_shape)

    print("\nEmbedding shapes for all models:")
    for model_name, shape in embedding_shape.items():
        print(f"{model_name}: {shape}")

def save_embeddings_for_model(audio_folder_path: str, model_name: str, embedding_shape: dict):
    """Compute and save embeddings for all wav files in a folder using a specified model."""
    model = Speaker(model_name)
    model_name = Path(model_name).parts[-2]
    os.makedirs(os.path.join("embeddings", model_name), exist_ok=True)
    for lang in LANGUAGE_ORDER:
        for speaker in SPEAKER_ORDER:
            wav_file_path_1 = os.path.join(audio_folder_path, lang, f"{speaker}_{lang}_utt1.wav")
            wav_file_path_2 = os.path.join(audio_folder_path, lang, f"{speaker}_{lang}_utt2.wav")
            # Load files (librosa converts them to floating-point arrays)
            data1, sr1 = librosa.load(wav_file_path_1)
            data2, sr2 = librosa.load(wav_file_path_2)

            # Ensure they have the same sample rate, then concatenate
            if sr1 == sr2:
                combined = np.concatenate((data1, data2))
                sf.write("joined_file.wav", combined, sr1)

            embedding = model.extract_embedding("joined_file.wav")
            embedding_info = {
                'speaker': speaker,
                'language': lang,
                'utterance': "utt_1_2",
                'embedding': embedding.cpu().numpy()  # Convert to numpy array for saving
            }
            
            save_path = os.path.join("embeddings", model_name, f"{speaker}_{lang}_embedding.pt")
            torch.save(embedding_info, save_path)
            print(f"Saved embeddings to {save_path}")
    embedding_shape[model_name] = embedding.shape
    print(model_name, embedding.shape)


def main():
    audio_folder = '/home/xiyali/git/wespeaker/stim'
    
    compute_similarity_matrices(audio_folder)

    save_embeddings(audio_folder)

if __name__ == '__main__':
    main()