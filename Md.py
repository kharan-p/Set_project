

# ============================================================
# RMSD COMPARISON: REFERENCE vs COMPOUND
# Reference = BLACK | Compound = RED
# Google Drive -> Google Colab
# Output: 600 DPI PNG + PDF
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
from google.colab import drive

# 1. Mount Google Drive
drive.mount('/content/drive')

# 2. Set your folder
FOLDER = '/content/drive/MyDrive/MD'

REF_FILE = os.path.join(FOLDER, 'RMSD ref.csv')
CMP_FILE = os.path.join(FOLDER, 'RMSD.csv')

# 3. Check files
for path in [REF_FILE, CMP_FILE]:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"File not found: {path}\n"
            "Check your Google Drive MD folder and filenames."
        )

# 4. Read data from Google Drive
ref = pd.read_csv(REF_FILE)
cmp = pd.read_csv(CMP_FILE)

ref.columns = ref.columns.astype(str).str.strip()
cmp.columns = cmp.columns.astype(str).str.strip()

print("Reference columns:", ref.columns.tolist())
print("Compound columns:", cmp.columns.tolist())

# 5. Expected column names
time_col = 'Time (ns)'
rmsd_col = 'RMSD (Å)'

for label, df in [('Reference', ref), ('Compound', cmp)]:
    if time_col not in df.columns or rmsd_col not in df.columns:
        raise ValueError(
            f"{label} CSV has unexpected column names.\n"
            f"Available columns: {df.columns.tolist()}\n"
            f"Expected: '{time_col}' and '{rmsd_col}'"
        )

# 6. Clean data
def prepare_data(df):
    df = df[[time_col, rmsd_col]].copy()
    df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
    df[rmsd_col] = pd.to_numeric(df[rmsd_col], errors='coerce')
    return df.dropna().sort_values(time_col)

ref = prepare_data(ref)
cmp = prepare_data(cmp)

if ref.empty or cmp.empty:
    raise ValueError("No valid numeric RMSD data found.")

# 7. Publication-style plot
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'axes.linewidth': 1.1
})

fig, ax = plt.subplots(figsize=(10, 5.5))

# REFERENCE = BLACK
ax.plot(
    ref[time_col],
    ref[rmsd_col],
    color='black',
    label='Reference protein',
    linewidth=1.4
)

# COMPOUND = RED
ax.plot(
    cmp[time_col],
    cmp[rmsd_col],
    color='red',
    label='Compound-bound complex',
    linewidth=1.4
)

ax.set_xlabel('Time (ns)')
ax.set_ylabel('RMSD (Å)')
ax.set_title('RMSD Comparison: Reference vs Compound')

ax.legend(frameon=False)
ax.grid(True, linestyle='--', alpha=0.25)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.tight_layout()

# 8. Save outputs directly to Google Drive
PNG_FILE = os.path.join(FOLDER, 'RMSD_Comparison.png')
PDF_FILE = os.path.join(FOLDER, 'RMSD_Comparison.pdf')

fig.savefig(PNG_FILE, dpi=600, bbox_inches='tight')
fig.savefig(PDF_FILE, bbox_inches='tight')

plt.show()
plt.close(fig)

# 9. Numerical summary
print('\n========== RMSD SUMMARY ==========')
print(f"Reference points: {len(ref)}")
print(f"Compound points:  {len(cmp)}")
print(f"Reference mean RMSD: {ref[rmsd_col].mean():.4f} Å")
print(f"Compound mean RMSD:  {cmp[rmsd_col].mean():.4f} Å")
print(f"Reference SD: {ref[rmsd_col].std():.4f} Å")
print(f"Compound SD:  {cmp[rmsd_col].std():.4f} Å")

print('\nSaved successfully to:')
print(PNG_FILE)
print(PDF_FILE)



# ============================================================
# RMSF COMPARISON: REFERENCE vs COMPOUND
# Reference = BLACK | Compound = RED
# Google Drive -> Google Colab
# Output: 600 DPI PNG + PDF
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
from google.colab import drive

# 1. Mount Google Drive
drive.mount('/content/drive')

# 2. Folder containing your CSV files
FOLDER = '/content/drive/MyDrive/MD'

REF_FILE = os.path.join(FOLDER, 'RMSF ref.csv')
CMP_FILE = os.path.join(FOLDER, 'RMSF.csv')

# 3. Check files
for path in [REF_FILE, CMP_FILE]:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"File not found: {path}\n"
            "Check the filenames inside Google Drive > MyDrive > MD."
        )

# 4. Read files directly from Google Drive
ref = pd.read_csv(REF_FILE)
cmp = pd.read_csv(CMP_FILE)

ref.columns = ref.columns.astype(str).str.strip()
cmp.columns = cmp.columns.astype(str).str.strip()

print("Reference columns:", ref.columns.tolist())
print("Compound columns:", cmp.columns.tolist())

# 5. Expected column names
res_col = 'Residue Number'
rmsf_col = 'RMSF (Å)'

for label, df in [('Reference', ref), ('Compound', cmp)]:
    if res_col not in df.columns or rmsf_col not in df.columns:
        raise ValueError(
            f"{label} CSV has unexpected column names.\n"
            f"Available columns: {df.columns.tolist()}\n"
            f"Expected: '{res_col}' and '{rmsf_col}'"
        )

# 6. Clean and prepare data
def prepare_data(df):
    df = df[[res_col, rmsf_col]].copy()
    df[res_col] = pd.to_numeric(df[res_col], errors='coerce')
    df[rmsf_col] = pd.to_numeric(df[rmsf_col], errors='coerce')
    return df.dropna().sort_values(res_col)

ref = prepare_data(ref)
cmp = prepare_data(cmp)

if ref.empty or cmp.empty:
    raise ValueError("No valid numeric RMSF data found.")

# 7. Plot both datasets on the same graph
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'axes.linewidth': 1.1
})

fig, ax = plt.subplots(figsize=(11, 5.5))

# REFERENCE = BLACK
ax.plot(
    ref[res_col],
    ref[rmsf_col],
    color='black',
    label='Reference protein',
    linewidth=1.3
)

# COMPOUND = RED
ax.plot(
    cmp[res_col],
    cmp[rmsf_col],
    color='red',
    label='Compound-bound complex',
    linewidth=1.3
)

ax.set_xlabel('Residue Number')
ax.set_ylabel('RMSF (Å)')
ax.set_title('RMSF Comparison: Reference vs Compound')

ax.legend(frameon=False)
ax.grid(True, linestyle='--', alpha=0.25)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.tight_layout()

# 8. Save directly to Google Drive
PNG_FILE = os.path.join(FOLDER, 'RMSF_Comparison.png')
PDF_FILE = os.path.join(FOLDER, 'RMSF_Comparison.pdf')

fig.savefig(PNG_FILE, dpi=600, bbox_inches='tight')
fig.savefig(PDF_FILE, bbox_inches='tight')

plt.show()
plt.close(fig)

# 9. Numerical summary
print('\n========== RMSF SUMMARY ==========')
print(f"Reference residues: {len(ref)}")
print(f"Compound residues:  {len(cmp)}")
print(f"Reference mean RMSF: {ref[rmsf_col].mean():.4f} Å")
print(f"Compound mean RMSF:  {cmp[rmsf_col].mean():.4f} Å")
print(f"Reference SD: {ref[rmsf_col].std():.4f} Å")
print(f"Compound SD:  {cmp[rmsf_col].std():.4f} Å")

print('\nSaved to Google Drive:')
print(PNG_FILE)
print(PDF_FILE)



# ============================================================
# RADIUS OF GYRATION (Rg) COMPARISON
# Reference protein: BLACK
# Compound-bound complex: RED
# Remove Rg values > 200 Å from BOTH datasets
# Save plots as 600 DPI PNG and PDF
# ============================================================

from google.colab import drive
drive.mount('/content/drive')

import os
import pandas as pd
import matplotlib.pyplot as plt

# 1. FILE PATHS
folder = "/content/drive/MyDrive/MD"

ref_file = os.path.join(folder, "RG ref.csv")
complex_file = os.path.join(folder, "RG.csv")

# 2. LOAD DATA
ref = pd.read_csv(ref_file)
complex_df = pd.read_csv(complex_file)

print("Reference columns:", ref.columns.tolist())
print("Complex columns:", complex_df.columns.tolist())

# 3. IDENTIFY TIME AND Rg COLUMNS
def find_column(df, column_type):
    for col in df.columns:
        normalized = (
            str(col).lower()
            .replace(" ", "")
            .replace("å", "a")
            .replace("Å", "a")
            .replace("_", "")
        )

        if column_type == "time" and "time" in normalized:
            return col

        if column_type == "rg":
            if "rg" in normalized or "radiusofgyration" in normalized:
                return col

    raise ValueError(
        f"Cannot identify the {column_type} column. "
        f"Available columns: {df.columns.tolist()}"
    )

ref_time_col = find_column(ref, "time")
ref_rg_col = find_column(ref, "rg")

complex_time_col = find_column(complex_df, "time")
complex_rg_col = find_column(complex_df, "rg")

print("\nReference time column:", ref_time_col)
print("Reference Rg column:", ref_rg_col)
print("Complex time column:", complex_time_col)
print("Complex Rg column:", complex_rg_col)

# 4. CONVERT DATA TO NUMERIC
for df, time_col, rg_col in [
    (ref, ref_time_col, ref_rg_col),
    (complex_df, complex_time_col, complex_rg_col)
]:
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df[rg_col] = pd.to_numeric(df[rg_col], errors="coerce")
    df.dropna(subset=[time_col, rg_col], inplace=True)

# 5. REMOVE Rg VALUES ABOVE 200 Å FROM BOTH DATASETS
ref_before = len(ref)
complex_before = len(complex_df)

ref = ref[ref[ref_rg_col] <= 200].copy()
complex_df = complex_df[complex_df[complex_rg_col] <= 200].copy()

print("\n========== FILTERING REPORT ==========")
print(f"Reference points removed: {ref_before - len(ref)}")
print(f"Reference points retained: {len(ref)}")
print(f"Complex points removed: {complex_before - len(complex_df)}")
print(f"Complex points retained: {len(complex_df)}")

# 6. SORT BY TIME
ref = ref.sort_values(ref_time_col)
complex_df = complex_df.sort_values(complex_time_col)

# 7. PLOT
plt.figure(figsize=(12, 6))

plt.plot(
    ref[ref_time_col],
    ref[ref_rg_col],
    color="black",
    linewidth=1.0,
    label="Reference protein"
)

plt.plot(
    complex_df[complex_time_col],
    complex_df[complex_rg_col],
    color="red",
    linewidth=1.0,
    label="Compound-bound complex"
)

plt.xlabel("Time (ns)", fontsize=13)
plt.ylabel("Radius of Gyration (Å)", fontsize=13)
plt.title("Radius of Gyration (Rg): Reference vs Compound", fontsize=15)

plt.legend(frameon=False, fontsize=11)
plt.grid(True, linestyle="--", alpha=0.25)
plt.tight_layout()

# 8. SAVE PLOTS TO GOOGLE DRIVE
png_path = os.path.join(folder, "Rg_Comparison.png")
pdf_path = os.path.join(folder, "Rg_Comparison.pdf")

plt.savefig(png_path, dpi=600, bbox_inches="tight")
plt.savefig(pdf_path, format="pdf", bbox_inches="tight")

plt.show()

print("\n========== OUTPUT FILES ==========")
print("PNG:", png_path)
print("PDF:", pdf_path)

# 9. SUMMARY STATISTICS
print("\n========== REFERENCE PROTEIN ==========")
print(ref[ref_rg_col].describe())

print("\n========== COMPOUND-BOUND COMPLEX ==========")
print(complex_df[complex_rg_col].describe())

# 10. VERIFY FILTERING
if len(ref) > 0:
    print("\nMaximum reference Rg after filtering:",
          ref[ref_rg_col].max(), "Å")

if len(complex_df) > 0:
    print("Maximum complex Rg after filtering:",
          complex_df[complex_rg_col].max(), "Å")

print("\nCompleted successfully.")
