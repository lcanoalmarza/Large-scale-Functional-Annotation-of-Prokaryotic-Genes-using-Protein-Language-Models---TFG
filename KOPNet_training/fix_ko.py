import numpy as np
import pandas as pd
print('Dependencies loaded')
# 1. Cargar el fichero prokaryotes.clean.dat y crear diccionario {ID: KO}

def normalize_id(seq_id):
    return seq_id.replace(".", "_").rstrip().split(" ")[0]

ko_dict = {}
with open("/home/lcano/projects/KEGG_pro_embeddings/prokaryotes.clean.dat", "r") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            norm_id = normalize_id(parts[0])
            ko_dict[norm_id] = parts[1]

print(f"Cargados {len(ko_dict)} IDs con KO desde prokaryotes.clean.dat")

# 2. Cargar datos UMAP y IDs (ajusta las rutas y variables según tu código)
data = np.load("13M_mapping.npz", allow_pickle=True)
# data debe tener 'coordinates' y 'ids'
coordinates = data['coordinates']
ids = data['ids']

print(f"Cargadas {coordinates.shape[0]} coordenadas UMAP y {len(ids)} IDs")

# 3. Alinear los KO con los IDs (manejando si los IDs vienen como bytes)
ko_alineados = [ko_dict.get(id_.decode() if isinstance(id_, bytes) else id_, "NA") for id_ in ids]

# 4. Crear DataFrame con todo junto
df = pd.DataFrame({
    "id": [id_.decode() if isinstance(id_, bytes) else id_ for id_ in ids],
    "ko": ko_alineados,
})
# Añadimos las coordenadas como columnas separadas (por ej, si son 2D o 3D)
for dim in range(coordinates.shape[1]):
    df[f"UMAP_{dim+1}"] = coordinates[:, dim]

print(df.head())

# 5. Guardar resultado para uso futuro (opcional)
df.to_csv("aligned_umap_ko.csv", index=False)
print("Datos alineados guardados en aligned_umap_ko.csv")
# 6. Guardar solo las anotaciones KO (en el mismo orden que UMAP)
ko_array = np.array(ko_alineados)
np.save("labels_ko.npy", ko_array)  # Opción binaria (rápida)
np.savetxt("labels_ko.csv", ko_array, fmt="%s")  # Opción en texto CSV

print("Etiquetas KO guardadas en labels_ko.csv y labels_ko.npy")
