from preprocessing import load_face_dataset

training, testing = load_face_dataset()

print(f"Training shape: {training.shape}")
print(f"Testing shape:  {testing.shape}")

assert training.shape == (600, 16384), "Wrong training size!"
assert testing.shape  == (150, 16384), "Wrong testing size!"