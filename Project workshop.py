import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QTextEdit, QComboBox,
                             QLabel, QFileDialog, QDoubleSpinBox, QSpinBox)
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

# Локализация колонок
COLUMN_RU = {
    "price_change": "Изменение цены", "total_value": "Общая стоимость",
    "destroyed_value": "Уничтожено", "mined_value": "Добыто",
    "produced_value": "Произведено", "trade_value": "Торговля",
    "total_isk": "Денежная масса", "isk_velocity": "Скорость ИСК",
    "npc_bounties": "Награды за NPC", "isk_lost": "Потеряно ИСК"
}
RU_TO_EN = {v: k for k, v in COLUMN_RU.items()}


class UniversalProcessor:
    def __init__(self):
        self.df = None
        self.working_df = None
        self.date_col = None
        self.cat_col = None

    def load(self, path):
        try:
            self.df = pd.read_csv(path)
            for col in self.df.columns:
                if any(x in col.lower() for x in ['date', 'time']):
                    self.df[col] = pd.to_datetime(self.df[col])
                    self.date_col = col
                    break
            priority_cats = ['sub_index', 'region_name', 'security_band', 'group_name']
            self.cat_col = next((c for c in priority_cats if c in self.df.columns), None)
            if not self.cat_col:
                for col in self.df.select_dtypes(include=['object']).columns:
                    if self.df[col].nunique() < 200:
                        self.cat_col = col
                        break
            return f"✅ Загружено. Анализ по: {self.cat_col}"
        except Exception as e:
            return f"❌ Ошибка: {e}"

    def get_stats(self, data):
        m, med = data.mean(), data.median()
        return {
            "Среднее": m, "Медиана": med, "СКО": data.std(),
            "Дисперсия": data.var(), "Мин": data.min(), "Макс": data.max(),
            "MAD (ср.абс)": (data - m).abs().mean(),
            "MEDAD (мед.абс)": (data - med).abs().median(),
            "IQR": data.quantile(0.75) - data.quantile(0.25),
            "Асимметрия": data.skew(), "Эксцесс": data.kurtosis()
        }


class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.proc = UniversalProcessor()
        self.setWindowTitle("EVE Online: Универсальный Аналитический Комплекс")
        self.resize(1400, 950)

        w = QWidget();
        self.setCentralWidget(w)
        layout = QHBoxLayout(w)

        tools = QVBoxLayout()
        self.btn_load = QPushButton("📁Загрузить данные")
        self.btn_load.clicked.connect(self.on_load)

        self.cb_cat = QComboBox()
        self.cb_col1 = QComboBox()  # Основной
        self.cb_col2 = QComboBox()  # Для связи

        self.cb_plot = QComboBox()
        self.cb_plot.addItems(
            ["Гистограмма + Плотность", "Box Plot (IQR)", "Box Plot (Среднее/СКО)", "Scatter Plot (Корреляция)"])

        self.z_sp = QDoubleSpinBox();
        self.z_sp.setValue(3.0)
        self.btn_stat = QPushButton("Анализ и Статистика")
        self.btn_stat.clicked.connect(self.on_stat)

        self.lag_spin = QSpinBox();
        self.lag_spin.setRange(1, 12);
        self.lag_spin.setValue(3)
        self.steps = QSpinBox();
        self.steps.setValue(12)

        self.btn_pred = QPushButton("Построить прогноз")
        self.btn_pred.clicked.connect(self.on_pred)

        self.log_box = QTextEdit();
        self.log_box.setReadOnly(True)
        self.log_box.setStyleSheet("background-color: #121212; color: #00FF41; font-family: 'Consolas';")

        tools.addWidget(self.btn_load)
        tools.addWidget(QLabel("Группа (Sub-Index):"))
        tools.addWidget(self.cb_cat)
        tools.addWidget(QLabel("Показатель 1 (Основа):"))
        tools.addWidget(self.cb_col1)
        tools.addWidget(QLabel("Показатель 2 (Для связи):"))
        tools.addWidget(self.cb_col2)
        tools.addWidget(QLabel("Тип графика:"))
        tools.addWidget(self.cb_plot)
        tools.addWidget(QLabel("Z-порог очистки:"))
        tools.addWidget(self.z_sp)
        tools.addWidget(self.btn_stat)
        tools.addSpacing(15)
        tools.addWidget(QLabel("Глубина памяти (Лаги):"))
        tools.addWidget(self.lag_spin)
        tools.addWidget(QLabel("Месяцев прогноза:"))
        tools.addWidget(self.steps)
        tools.addWidget(self.btn_pred)
        tools.addWidget(self.log_box)

        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvas(self.fig)
        layout.addLayout(tools, 1);
        layout.addWidget(self.canvas, 2)

    def on_load(self):
        path, _ = QFileDialog.getOpenFileName(self, "Открыть", "", "CSV (*.csv)")
        if path:
            self.log_box.clear();
            self.log_box.append(self.proc.load(path))
            self.cb_cat.clear()
            if self.proc.cat_col:
                self.cb_cat.addItems(sorted([str(x) for x in self.proc.df[self.proc.cat_col].unique()]))
            cols_ru = [COLUMN_RU.get(c, c) for c in self.proc.df.select_dtypes(include=[np.number]).columns]
            self.cb_col1.clear();
            self.cb_col1.addItems(cols_ru)
            self.cb_col2.clear();
            self.cb_col2.addItems(cols_ru)

    def on_stat(self):
        try:
            cat = self.cb_cat.currentText()
            col1_ru = self.cb_col1.currentText()
            col2_ru = self.cb_col2.currentText()
            # Пытаемся найти английское имя, если не находим - берем как есть
            c1_eng = RU_TO_EN.get(col1_ru, col1_ru)
            c2_eng = RU_TO_EN.get(col2_ru, col2_ru)
            ptype = self.cb_plot.currentText()

            # --- УМНАЯ ФИЛЬТРАЦИЯ ---
            # Если есть колонка категорий и выбрано не "Все", фильтруем. Иначе берем весь DF.
            if self.proc.cat_col and cat and cat != "Все" and cat != "Все (без фильтра)":
                df_work = self.proc.df[self.proc.df[self.proc.cat_col] == cat].copy()
            else:
                df_work = self.proc.df.copy()

            # Базовая очистка от пустых значений в выбранной колонке
            df_work = df_work.dropna(subset=[c1_eng]).sort_values(self.proc.date_col)

            # Очистка Z-score (только если данных достаточно)
            if len(df_work) > 5:
                z = np.abs(stats.zscore(df_work[c1_eng]))
                df_work = df_work[z < self.z_sp.value()]

            self.proc.working_df = df_work

            # Вывод статистики
            self.log_box.append(f"\nАнализ: {cat if cat else 'Весь файл'} ({col1_ru})")
            s = self.proc.get_stats(df_work[c1_eng])
            for k, v in s.items():
                self.log_box.append(f" • {k}: {v:.4f}" if isinstance(v, float) else f" • {k}: {v}")

            self.ax.clear()
            d1 = df_work[c1_eng]

            if ptype == "Гистограмма + Плотность":
                self.ax.hist(d1, bins=20, alpha=0.6, color='green', density=True)
                kde = stats.gaussian_kde(d1);
                x = np.linspace(d1.min(), d1.max(), 100)
                self.ax.plot(x, kde(x), color='white', linewidth=2)
                self.ax.set_ylabel("Плотность вероятности")
            elif ptype == "Box Plot (IQR)":
                self.ax.boxplot(d1, vert=False, patch_artist=True, boxprops=dict(facecolor='cyan'))
            elif ptype == "Box Plot (Среднее/СКО)":
                m, sd = s["Среднее"], s["СКО"]
                self.ax.barh(1, 2 * sd, left=m - sd, height=0.3, color='magenta', alpha=0.4)
                self.ax.vlines(m, 0.7, 1.3, colors='yellow', linewidth=3)
            elif ptype == "Scatter Plot (Корреляция)":
                d2 = df_work[c2_eng]
                r = d1.corr(d2)
                self.ax.scatter(d1, d2, alpha=0.6, color='orange', edgecolors='white')
                self.ax.set_ylabel(col2_ru)
                self.log_box.append(f"\nКорреляция Пирсона: {r:.4f}")
                abs_r = abs(r)
                strength = "высокая" if abs_r > 0.7 else "умеренная" if abs_r > 0.3 else "слабая"
                self.log_box.append(f" • Сила связи: {strength}")

            self.ax.set_xlabel(col1_ru)
            self.ax.set_title(f"Анализ: {cat if cat else 'Данные'}", pad=20)
            self.fig.tight_layout();
            self.canvas.draw()
        except Exception as e:
            self.log_box.append(f"❌ Ошибка анализа: {e}")

    def on_pred(self):
        try:
            col_ru = self.cb_col1.currentText()
            col_eng = RU_TO_EN.get(col_ru, col_ru)
            n = self.steps.value()
            n_lags = self.lag_spin.value()

            df_work = self.proc.working_df
            if df_work is None or len(df_work) <= n_lags:
                self.log_box.append(f"⚠️ Мало данных для {n_lags} лагов.")
                return

            # Подготовка лагов
            data_lags = df_work[[col_eng]].copy()
            for i in range(1, n_lags + 1):
                data_lags[f'lag_{i}'] = data_lags[col_eng].shift(i)

            data_lags = data_lags.dropna()
            y = data_lags[col_eng].values
            X = data_lags.drop(columns=[col_eng]).values

            self.ax.clear();
            self.log_box.append(f"\nПрогноз ({col_ru}):")
            dates = pd.to_datetime(df_work[self.proc.date_col])

            v_idx = -24 if len(df_work) > 24 else 0
            self.ax.plot(dates.iloc[v_idx:], df_work[col_eng].iloc[v_idx:], label="Факт", color='white', linewidth=3,
                         marker='o', markersize=4)

            models = [("Линейная", LinearRegression(), 'yellow'),
                      ("Лес", RandomForestRegressor(n_estimators=100), 'cyan'),
                      ("Дерево", DecisionTreeRegressor(), 'magenta')]

            for name, m, color in models:
                m.fit(X, y)
                self.log_box.append(f" • {name} R²: {r2_score(y, m.predict(X)):.2f}")

                # Рекурсия
                curr = list(y[-n_lags:])[::-1]
                preds = []
                for _ in range(n):
                    p = m.predict(np.array(curr).reshape(1, -1))[0]
                    preds.append(p)
                    curr = [p] + curr[:-1]

                diff = dates.diff().median()
                fut_dates = [dates.iloc[-1] + (i + 1) * diff for i in range(n)]
                self.ax.plot([dates.iloc[-1]] + fut_dates, [y[-1]] + preds, '--', color=color, label=name)

            self.ax.set_xlim(dates.iloc[v_idx], fut_dates[-1])
            self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
            plt.setp(self.ax.get_xticklabels(), rotation=30, ha='right')
            self.ax.legend();
            self.fig.tight_layout();
            self.canvas.draw()
        except Exception as e:
            self.log_box.append(f"❌ Ошибка прогноза: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv);
    w = App();
    w.show();
    sys.exit(app.exec())

