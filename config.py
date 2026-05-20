import warnings
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = "data"   # director output (CSV + PNG)

N_TEST   = 12   # luni orizont prognoza
M_SEASON = 12   # sezonalitate lunara

COLORS = {
    "train":  "#2196F3",
    "test":   "#4CAF50",
    "hw":     "#FF9800",
    "sarima": "#F44336",
    "ci":     "#FFCDD2",
    "oil":    "#FF5722",
    "fx":     "#9C27B0",
}

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 100, "font.size": 10})
sns.set_theme(style="whitegrid", palette="muted")
