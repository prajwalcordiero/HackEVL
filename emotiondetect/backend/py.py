import h5py

filename = "emotion_model_full.keras"  # or "emotion_model_full.h5"

with h5py.File(filename, "r") as f:
    print("Keys in file:", list(f.keys()))
    if "model_weights" in f:
        print("✅ Model has weights stored")
        print("Layers inside:")
        for layer_name in f["model_weights"].keys():
            print(" -", layer_name)
    else:
        print("⚠️ No model_weights found, file may be empty")
