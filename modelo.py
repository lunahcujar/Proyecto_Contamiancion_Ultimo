import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

def run_analysis():
    os.makedirs("static", exist_ok=True)

    df = pd.read_csv("Datos_enriquecido_final(1)(1).csv", index_col=0, parse_dates=True).dropna()
    df.columns = df.columns.str.strip()
    df = df.rename(columns=lambda x: x.upper())

    target = "PM25"
    targets = ["CO", "PM10"]

    # MATRIZ DE CORRELACIÓN REDUCIDA
    correlation_matrix = df[[target] + targets].corr()
    print("\n📊 MATRIZ DE CORRELACIÓN:")
    print(correlation_matrix.round(2))

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Matriz de Correlación entre PM25, CO y PM10")
    plt.tight_layout()
    plt.savefig("static/heatmap_correlacion.png")
    plt.close()

    r2_scores = {}

    # REGRESIONES UNIVARIADAS Y SUS DISPERSIONES
    for t in targets:
        # Dispersión original
        plt.figure(figsize=(6, 4))
        sns.scatterplot(x=df[target], y=df[t], alpha=0.6)
        sns.regplot(x=df[target], y=df[t], scatter=False, color="red")
        plt.xlabel("PM25")
        plt.ylabel(t)
        plt.title(f"Dispersión PM25 vs {t}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"static/dispersion_pm25_vs_{t.lower()}.png")
        plt.close()

        # Regresión univariada
        X = df[[target]]
        y = df[t]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.05, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = round(r2_score(y_test, y_pred), 4)
        r2_scores[f"{t}_univariada"] = r2

        # Gráfico regresión
        plt.figure(figsize=(6, 4))
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
        sns.lineplot(x=y_test, y=y_test, color="red", label="Ideal")
        plt.xlabel(f"{t} Real")
        plt.ylabel(f"{t} Predicho")
        plt.title(f"Regresión PM25 → {t} (R² = {r2:.2f})")
        plt.grid(True)
        plt.tight_layout()
        plt.legend()
        plt.savefig(f"static/regresion_{t.lower()}.png")
        plt.close()

        # Dispersión real vs predicho
        plt.figure(figsize=(6, 4))
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
        plt.xlabel(f"{t} Real")
        plt.ylabel(f"{t} Predicho")
        plt.title(f"Dispersión de Regresión PM25 → {t}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"static/dispersion_regresion_pm25_{t.lower()}.png")
        plt.close()

    # REGRESIÓN MULTIVARIADA PARA PM10
    print("\n🔍 REGRESIÓN MULTIVARIADA PARA PM10:")
    X_full = df[[col for col in df.columns if col != "PM10"]]
    y_full = df["PM10"]

    X_scaled = StandardScaler().fit_transform(X_full)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_full, test_size=0.2, random_state=42)

    model_multi = XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.05, random_state=42)
    model_multi.fit(X_train, y_train)
    y_pred_multi = model_multi.predict(X_test)

    r2_multi = round(r2_score(y_test, y_pred_multi), 4)
    r2_scores["PM10_multivariada"] = r2_multi
    print(f"✅ PM10 Multivariada: R² = {r2_multi}")

    # Gráfico regresión multivariada
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=y_test, y=y_pred_multi, alpha=0.6)
    sns.lineplot(x=y_test, y=y_test, color="red", label="Ideal")
    plt.xlabel("PM10 Real")
    plt.ylabel("PM10 Predicho")
    plt.title(f"Regresión Multivariada PM10 (R² = {r2_multi})")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.savefig("static/regresion_pm10_multivariada.png")
    plt.close()

    # Dispersión multivariada PM10
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=y_test, y=y_pred_multi, alpha=0.6)
    plt.xlabel("PM10 Real")
    plt.ylabel("PM10 Predicho")
    plt.title("Dispersión PM10 Multivariada")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("static/dispersion_pm10_multivariada.png")
    plt.close()

    # PROMEDIO ANUAL DE PM25
    yearly_avg = df.resample("YE").mean()["PM25"]
    print("\n📆 PROMEDIO ANUAL DE PM25:")
    print(yearly_avg.round(2))

    plt.figure(figsize=(10, 5))
    yearly_avg.plot(kind="bar", color="orange")
    plt.title("Promedio Anual de PM25")
    plt.xlabel("Año")
    plt.ylabel("PM25")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(True)
    plt.savefig("static/pm25_anual.png")
    plt.close()

    return {
        "correlation_matrix": correlation_matrix.round(2).to_dict(),
        "r2_scores": r2_scores,
        "yearly_avg": yearly_avg.round(2).to_dict()
    }

if __name__ == "__main__":
    run_analysis()
