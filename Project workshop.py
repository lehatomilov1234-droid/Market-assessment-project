import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QTextEdit, QComboBox,
                             QLabel, QFileDialog, QDoubleSpinBox, QSpinBox)
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

# Локализация колонок EVE Online
COLUMN_RU = {
    "price_change": "Изменение цены", "total_value": "Общая стоимость",
    "destroyed_value": "Уничтожено (ISK)", "mined_value": "Добыто (ISK)",
    "produced_value": "Произведено (ISK)", "trade_value": "Объем торгов (ISK)",
    "total_isk": "Денежная масса", "isk_velocity": "Скорость обращения",
    "asteroid_volume_mined": "Добыча руды", "gas_volume_mined": "Добыча газа",
    "ice_volume_mined": "Добыча льда", "moon_volume_mined": "Добыча лун",
    "npc_bounties": "Баунти (NPC)"
}
RU_TO_EN = {v: k for k, v in COLUMN_RU.items()}


class EveProcessor:
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
                    self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                    self.date_col = col
                    break
            self.cat_col = None
            for col in self.df.select_dtypes(include=['object']).columns:
                if self.df[col].nunique() < 100:
                    self.cat_col = col
                    break
            return f"✅ Загружено строк: {len(self.df)}"
        except Exception as e:
            return f"❌ Ошибка: {e}"

    def clean(self, cat_val, z_thresh):
        data = self.df[self.df[self.cat_col] == cat_val].copy() if self.cat_col and cat_val != "Все" else self.df.copy()
        data = data.dropna().drop_duplicates()
        nums = data.select_dtypes(include=[np.number]).columns
        if len(data) > 10:
            z = np.abs(stats.zscore(data[nums]))
            data = data[(z < z_thresh).all(axis=1)]
        self.working_df = data.sort_values(self.date_col) if self.date_col else data
        return len(self.working_df)

    def get_stats(self, col_ru):
        col = RU_TO_EN.get(col_ru, col_ru)
        d = self.working_df[col]
        m, med = d.mean(), d.median()
        return {
            "Среднее": m, "Медиана": med, "Мода": d.mode()[0] if not d.mode().empty else 0,
            "Мин": d.min(), "Макс": d.max(), "Дисперсия": d.var(), "СКО": d.std(),
            "MAD (ср.абс)": (d - m).abs().mean(), "MEDAD (мед.абс)": (d - med).abs().median(),
            "IQR": d.quantile(0.75) - d.quantile(0.25), "Асимметрия": d.skew(), "Эксцесс": d.kurtosis()
        }


class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.proc = EveProcessor()
        self.setWindowTitle("EVE Online: Универсальный комплекс прогнозирования")
        self.resize(1300, 900)

        widget = QWidget()
        self.setCentralWidget(widget)
        layout = QHBoxLayout(widget)

        # ПАНЕЛЬ УПРАВЛЕНИЯ
        tools = QVBoxLayout()
        self.btn_load = QPushButton("📁 1. Загрузить данные")
        self.btn_load.clicked.connect(self.on_load)

        self.cb_cat = QComboBox()
        self.cb_col1 = QComboBox()  # Основная колонка
        self.cb_col2 = QComboBox()  # Вторая колонка для корреляции

        self.cb_plot = QComboBox()
        self.cb_plot.addItems(
            ["Гистограмма + Плотность", "Box Plot (IQR)", "Box Plot (Среднее/СКО)", "Scatter Plot (Корреляция)"])

        self.z_sp = QDoubleSpinBox()
        self.z_sp.setValue(3.0)

        self.btn_stat = QPushButton("📊 2. Анализ и Статистика")
        self.btn_stat.clicked.connect(self.on_stat)

        self.steps = QSpinBox()
        self.steps.setValue(12)
        self.btn_pred = QPushButton("🔮 3. Сравнить 3 модели прогноза")
        self.btn_pred.clicked.connect(self.on_pred)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setStyleSheet("background-color: #121212; color: #00FF41; font-family: 'Consolas';")

        tools.addWidget(self.btn_load)
        tools.addWidget(QLabel("Категория:"))
        tools.addWidget(self.cb_cat)
        tools.addWidget(QLabel("Показатель 1 (Основа):"))
        tools.addWidget(self.cb_col1)
        tools.addWidget(QLabel("Показатель 2 (Для связи):"))
        tools.addWidget(self.cb_col2)
        tools.addWidget(QLabel("Тип графика:"))
        tools.addWidget(self.cb_plot)
        tools.addWidget(QLabel("Z-порог:"))
        tools.addWidget(self.z_sp)
        tools.addWidget(self.btn_stat)
        tools.addSpacing(15)
        tools.addWidget(QLabel("Шагов прогноза:"))
        tools.addWidget(self.steps)
        tools.addWidget(self.btn_pred)
        tools.addWidget(self.log_box)

        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvas(self.fig)

        layout.addLayout(tools, 1)
        layout.addWidget(self.canvas, 2)

    def on_load(self):
        path, _ = QFileDialog.getOpenFileName(self, "Открыть", "", "CSV (*.csv)")
        if path:
            self.log_box.clear()
            self.log_box.append(self.proc.load(path))
            self.cb_cat.clear()
            self.cb_cat.addItem("Все")
            if self.proc.cat_col:
                self.cb_cat.addItems(sorted([str(x) for x in self.proc.df[self.proc.cat_col].unique()]))
            cols = [COLUMN_RU.get(c, c) for c in self.proc.df.select_dtypes(include=[np.number]).columns]
            self.cb_col1.clear();
            self.cb_col1.addItems(cols)
            self.cb_col2.clear();
            self.cb_col2.addItems(cols)

    def on_stat(self):
        col1, col2 = self.cb_col1.currentText(), self.cb_col2.currentText()
        count = self.proc.clean(self.cb_cat.currentText(), self.z_sp.value())
        self.log_box.append(f"\n✅ Очистка: доступно {count} строк.")

        # Вывод статистики для первой колонки
        s = self.proc.get_stats(col1)
        self.log_box.append(f"📈 Статистика ({col1}):")
        for k, v in s.items(): self.log_box.append(f" • {k}: {v:.4f}" if isinstance(v, float) else f" • {k}: {v}")

        self.ax.clear()
        ptype = self.cb_plot.currentText()
        d1 = self.proc.working_df[RU_TO_EN.get(col1, col1)]

        # --- ГИСТОГРАММА + ПЛОТНОСТЬ ---
        if ptype == "Гистограмма + Плотность":
            # Рисуем гистограмму
            self.ax.hist(d1, bins=25, alpha=0.5, color='lime', density=True, label='Распределение (Гистограмма)')

            # Рисуем линию плотности
            kde = stats.gaussian_kde(d1)
            x = np.linspace(d1.min(), d1.max(), 100)
            self.ax.plot(x, kde(x), color='white', linewidth=2, label='Кривая плотности (KDE)')

            # Добавляем информативные надписи (ОБНОВЛЕНО)
            self.ax.set_title(f"Анализ распределения: {col1}", fontsize=12, pad=15)
            self.ax.set_xlabel(f"Значение показателя ({col1})", fontsize=10)
            self.ax.set_ylabel("Плотность вероятности", fontsize=10)
            self.ax.grid(True, linestyle='--', alpha=0.3)  # Сетка для удобства
            self.ax.legend()

        # --- BOX PLOT IQR ---
        elif ptype == "Box Plot (IQR)":
            self.ax.boxplot(d1, vert=False, patch_artist=True, boxprops=dict(facecolor='cyan', alpha=0.6))
            self.ax.set_title(f"Диаграмма размаха (Медиана/IQR): {col1}")
            self.ax.set_xlabel("Значение")
            self.ax.set_yticks([])  # Убираем лишние деления по Y

        # --- BOX PLOT MEAN/SD ---
        elif ptype == "Box Plot (Среднее/СКО)":
            m, sd = s["Среднее"], s["СКО"]
            self.ax.barh(1, 2 * sd, left=m - sd, height=0.3, color='magenta', alpha=0.4,
                         label='Граница ±1 СКО (68% данных)')
            self.ax.vlines(m, 0.7, 1.3, colors='yellow', linewidth=3, label=f'Среднее: {m:.2f}')
            self.ax.hlines(1, d1.min(), d1.max(), colors='white', alpha=0.5, label='Минимум/Максимум')
            self.ax.set_title(f"Разброс данных (Среднее и СКО): {col1}")
            self.ax.set_xlabel("Значение")
            self.ax.set_yticks([])
            self.ax.legend()

        # --- SCATTER PLOT ---
        elif ptype == "Scatter Plot (Корреляция)":
            d2 = self.proc.working_df[RU_TO_EN.get(col2, col2)]
            r = d1.corr(d2)

            # Определение силы связи (Шкала Чеддока)
            abs_r = abs(r)
            if abs_r < 0.1:
                strength = "практически отсутствует"
            elif abs_r < 0.3:
                strength = "слабая"
            elif abs_r < 0.5:
                strength = "умеренная"
            elif abs_r < 0.7:
                strength = "заметная"
            elif abs_r < 0.9:
                strength = "высокая"
            else:
                strength = "очень сильная"

            # Вывод в лог (только коэффициент и сила)
            self.log_box.append(f"\n🔗 КОРРЕЛЯЦИЯ ПИРСОНА:")
            self.log_box.append(f" • Коэффициент r: {r:.4f}")
            self.log_box.append(f" • Сила связи: {strength}")

            # Отрисовка графика рассеяния
            self.ax.scatter(d1, d2, alpha=0.6, color='orange', edgecolors='white')
            self.ax.set_title(f"Корреляция: r = {r:.2f}")
            self.ax.set_xlabel(col1)
            self.ax.set_ylabel(col2)
            self.ax.grid(True, linestyle=':', alpha=0.4)

        self.canvas.draw()

    def on_pred(self):
        col = RU_TO_EN.get(self.cb_col1.currentText(), self.cb_col1.currentText())
        n = self.steps.value()
        y = self.proc.working_df[col].values
        X = np.arange(len(y)).reshape(-1, 1)
        xf = np.arange(len(y), len(y) + n).reshape(-1, 1)

        self.log_box.append("\n🚀 Сравнение моделей:")
        self.ax.clear()
        self.ax.plot(y, label="Факт", color='white', alpha=0.4)

        models = [
            ("Линейная", LinearRegression(), 'yellow'),
            ("Случайный Лес", RandomForestRegressor(n_estimators=50), 'cyan'),
            ("Дерево", DecisionTreeRegressor(), 'magenta')
        ]

        for name, m, c in models:
            m.fit(X, y)
            r2 = r2_score(y, m.predict(X))
            self.log_box.append(f" • {name}: R² = {r2:.2f}")
            self.ax.plot(xf, m.predict(xf), '--', color=c, label=name)

        self.ax.legend();
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv);
    w = App();
    w.show();
    sys.exit(app.exec())
