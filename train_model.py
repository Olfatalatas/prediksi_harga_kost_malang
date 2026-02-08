import pandas as pd
import joblib
import optuna
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, classification_report, confusion_matrix

# Variabel Global untuk menampung teks log
LOG_DATA = []

def log_print(text):
    """Mencetak ke terminal DAN menyimpannya ke memori untuk nanti ditulis ke txt"""
    print(text)
    LOG_DATA.append(text)

def latih_final_battle():
    log_print("========================================================")
    log_print("   PERTARUNGAN MODEL: LINEAR REGRESSION VS RANDOM FOREST")
    log_print("========================================================")

    # 1. Load Data
    df = pd.read_csv('data_kost_malang_clean.csv')
    df = pd.get_dummies(df, columns=['Daerah_Clean', 'Jenis Kost'], drop_first=True)
    X = df.drop(['Harga_Angka', 'Nama Kost'], axis=1)
    y = df['Harga_Angka']

    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Helper untuk kategori harga
    batas_bawah, batas_atas = 850000, 1500000
    labels = ["Ekonomis", "Standar", "Eksklusif"]
    
    def kategorikan(harga):
        if harga < batas_bawah: return "Ekonomis"
        elif harga < batas_atas: return "Standar"
        else: return "Eksklusif"

    y_test_kat = [kategorikan(h) for h in y_test]

    # ====================================================================
    # RONDE 1: LINEAR REGRESSION
    # ====================================================================
    log_print("\n[1] Melatih Linear Regression...")
    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)
    
    # Evaluasi LR di Data Test
    y_pred_lr = model_lr.predict(X_test)
    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    r2_lr = r2_score(y_test, y_pred_lr)
    mape_lr = np.mean(np.abs((y_test - y_pred_lr) / y_test)) * 100
    
    log_print(f"    -> MAE (Error Rupiah): Rp {int(mae_lr):,}")
    log_print(f"    -> R2 Score (Akurasi): {r2_lr:.2f}")
    log_print(f"    -> MAPE (Error %): {mape_lr:.2f}%")
    
    # Classification Report LR (Agar adil, LR juga kita cek kategorinya)
    y_pred_lr_kat = [kategorikan(h) for h in y_pred_lr]
    log_print("\n    --- Classification Report (Linear Regression) ---")
    log_print(classification_report(y_test_kat, y_pred_lr_kat, target_names=labels))

    # ====================================================================
    # RONDE 2: RANDOM FOREST (BAYESIAN TUNING)
    # ====================================================================
    log_print("\n[2] Tuning Random Forest dengan Optuna...")

    # Kita pakai print biasa untuk log optuna supaya file txt tidak penuh sampah log trial
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 5, 50),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 15),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'random_state': 42,
            'n_jobs': -1
        }
        model = RandomForestRegressor(**params)
        score = cross_val_score(model, X_train, y_train, cv=3, scoring='neg_mean_absolute_error').mean()
        return -score

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=15)

    log_print(f"    -> Parameter Terbaik RF: {study.best_params}")

    # Latih Final RF
    best_params = study.best_params
    model_rf = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
    model_rf.fit(X_train, y_train)
    
    # Evaluasi RF di Data Test
    y_pred_rf = model_rf.predict(X_test)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)
    mape_rf = np.mean(np.abs((y_test - y_pred_rf) / y_test)) * 100
    
    log_print(f"    -> MAE (Error Rupiah): Rp {int(mae_rf):,}")
    log_print(f"    -> R2 Score (Akurasi): {r2_rf:.2f}")
    log_print(f"    -> MAPE (Error %): {mape_rf:.2f}%")

    # Classification Report RF
    y_pred_rf_kat = [kategorikan(h) for h in y_pred_rf]
    log_print("\n    --- Classification Report (Random Forest) ---")
    log_print(classification_report(y_test_kat, y_pred_rf_kat, target_names=labels))

    # ====================================================================
    # KEPUTUSAN JUARA
    # ====================================================================
    log_print("\n========================================================")
    log_print("   HASIL AKHIR (FINAL VERDICT)")
    log_print("========================================================")
    
    juara_model = None
    nama_juara = ""
    y_pred_final = []
    
    if mae_rf < mae_lr:
        log_print("🏆 PEMENANG: RANDOM FOREST")
        log_print(f"   (Lebih akurat Rp {int(mae_lr - mae_rf):,} dibanding Linear Regression)")
        juara_model = model_rf
        nama_juara = "Random Forest"
        y_pred_final = y_pred_rf
    else:
        log_print("🏆 PEMENANG: LINEAR REGRESSION")
        juara_model = model_lr
        nama_juara = "Linear Regression"
        y_pred_final = y_pred_lr
        
    joblib.dump(juara_model, 'model_kost_terbaik.pkl')
    joblib.dump(X.columns, 'list_fitur.pkl')
    log_print(f"Model {nama_juara} telah disimpan ke file .pkl")

    # ====================================================================
    # SIMPAN OUTPUT KE TXT & GRAFIK
    # ====================================================================
    nama_folder = "hasil_evaluasi"
    if not os.path.exists(nama_folder):
        os.makedirs(nama_folder)
    
    # 1. Simpan Text Log
    path_txt = os.path.join(nama_folder, 'laporan_komparasi_model.txt')
    with open(path_txt, 'w', encoding='utf-8') as f:
        f.write('\n'.join(LOG_DATA))
    print(f"\n[SUKSES] Laporan teks lengkap disimpan di:\n   -> {path_txt}")

    # 2. Simpan Grafik Juara
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Grafik A: Actual vs Predicted
    sns.scatterplot(x=y_test, y=y_pred_final, ax=axes[0, 0], color='blue', alpha=0.6)
    min_val = min(y_test.min(), y_pred_final.min())
    max_val = max(y_test.max(), y_pred_final.max())
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--')
    axes[0, 0].set_title(f'Akurasi {nama_juara}')
    axes[0, 0].set_xlabel('Harga Asli')
    axes[0, 0].set_ylabel('Prediksi AI')

    # Grafik B: Residual Plot
    residuals = y_test - y_pred_final
    sns.scatterplot(x=y_test, y=residuals, ax=axes[0, 1], color='orange', alpha=0.6)
    axes[0, 1].axhline(0, color='red', linestyle='--')
    axes[0, 1].set_title('Residuals: Distribusi Error')

    # Grafik C: Feature Importance / Koefisien
    if hasattr(juara_model, 'feature_importances_'):
        importances = juara_model.feature_importances_
        fi_df = pd.DataFrame({'Fitur': X.columns, 'Pentingnya': importances})
        fi_df = fi_df.sort_values(by='Pentingnya', ascending=False).head(10)
        sns.barplot(x='Pentingnya', y='Fitur', data=fi_df, ax=axes[1, 0], palette='viridis')
        axes[1, 0].set_title('Top 10 Faktor Penentu Harga')
    else:
        coefs = pd.DataFrame({'Fitur': X.columns, 'Koefisien': juara_model.coef_})
        coefs['Abs_Koefisien'] = coefs['Koefisien'].abs()
        coefs = coefs.sort_values(by='Abs_Koefisien', ascending=False).head(10)
        sns.barplot(x='Koefisien', y='Fitur', data=coefs, ax=axes[1, 0], palette='coolwarm')
        axes[1, 0].set_title('Top 10 Koefisien')

    # Grafik D: Confusion Matrix
    y_pred_kat = [kategorikan(h) for h in y_pred_final]
    cm = confusion_matrix(y_test_kat, y_pred_kat, labels=labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=axes[1, 1])
    axes[1, 1].set_title('Confusion Matrix Juara')

    plt.tight_layout()
    path_img = os.path.join(nama_folder, f"Grafik_{nama_juara.replace(' ', '_')}.png")
    plt.savefig(path_img, dpi=300)
    plt.close()
    print(f"[SUKSES] Grafik evaluasi disimpan di:\n   -> {path_img}")

if __name__ == "__main__":
    latih_final_battle()