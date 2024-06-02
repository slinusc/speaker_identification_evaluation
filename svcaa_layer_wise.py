import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
import os
import glob
from SVCCA import SVCCA

# Funktion zum Laden der Hidden States und Labels aus npy Dateien
def load_hidden_states_and_labels(root_dir, speakers, layer, max_files_per_speaker=50):
    all_hidden_states = []
    all_labels = []
    
    for speaker in tqdm(speakers):
        layer_path = os.path.join(root_dir, speaker, f'layer_{layer}')
        file_paths = glob.glob(os.path.join(layer_path, '*.npy'))[:max_files_per_speaker]
        for file_path in file_paths:
            hidden_states = np.load(file_path)
            if hidden_states.shape[2] == 1024:  # Überprüfen der Dimensionen
                hidden_states = hidden_states.reshape(-1, 1024)  # Umformung zu (n, 1024)
                all_hidden_states.append(hidden_states)
                labels = np.array([speaker] * hidden_states.shape[0]) # Erstellung der Labels für jeden Hidden State
                all_labels.append(labels)
    
    combined_hidden_states = np.vstack(all_hidden_states)
    combined_labels = np.hstack(all_labels)
    
    return combined_hidden_states, combined_labels

# Beispielhafte Sprecher und Anzahl der Layer
root_dir = '/home/rag/experimental_trial/data/all_speakers_w2vec_28.05'  # Anpassung notwendig
speakers = ['speaker_' + str(i) for i in range(1, 51)]
num_layers = 25  # Beispielhafte Anzahl von Layern

results = []

encoder = OneHotEncoder(sparse_output=False)

for i in range(num_layers):
    print(f"Processing layer {i}")
    hidden_states_layer, labels_layer = load_hidden_states_and_labels(root_dir, speakers, i)
    
    # One-Hot-Encoding der Labels
    Ys = encoder.fit_transform(labels_layer.reshape(-1, 1))
    
    # SVCCA ausführen
    svcca = SVCCA(hidden_states=[hidden_states_layer], labels=[Ys], use_gpu=False)
    svcca.calculate_max_correlations()
    
    # Zwischenspeicherung der Ergebnisse
    max_corr = svcca.max_correlations[0]
    results.append({'layer': i, 'max_correlation': max_corr})

# Ergebnisse in eine CSV-Datei speichern
results_df = pd.DataFrame(results)
results_df.to_csv('results/svcca_results_50_w2v.csv', index=False)
print("Ergebnisse erfolgreich in 'results/svcca_results_50_w2v.csv' gespeichert.")

# nohup python3 svcaa_layer_wise.py > output.log 2>&1 &