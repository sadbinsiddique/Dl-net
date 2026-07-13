from eda import *

def eda_imp(DATA_PATH: str, OUTPUT_DIR: str, NAME: str):
    counter = 1
    print("+" * 50)
    print(f"Step 02: [EDA]")
    
    print("+" * 50)
    print("[1]Starting EDA :", DATA_PATH)

    eda = ImageDatasetEDA(DATA_PATH)
    eda.validate_and_extract_metadata()

    if eda.df.empty:
        print("No data")
        return

    eda.analyze_distribution()
    eda.df = run_quality_pipeline(eda.df)
    duplicates = eda.df.duplicated(eda.df)
    mean_face, _ = calculate_ssim_outliers(eda.df)

    plt.imshow(mean_face, cmap='gray') 
    plt.title('Dataset Mean face')
    plt.axis('off')
    plt.show()

    face_landmarks = analyze_faces_mediapipe(eda.df)
    if face_landmarks is not None and not face_landmarks.empty:
        face_landmarks.to_csv(os.path.join(OUTPUT_DIR, "face_landmarks.csv"), index=False)
        print(f"[11]Face landmarks saved successfully.")
    visualize_embeddings(eda.df)

    eda.df.to_csv(os.path.join(OUTPUT_DIR, NAME), index=False)
    print(f"[8]{OUTPUT_DIR}/{NAME} saved successfully.")
    counter += 1

    return eda.df
