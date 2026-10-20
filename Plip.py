

# ============================================================
# PLIP INTERACTION FINGERPRINT
# A: CALYCOSIN
# B: FORSYTHOSIDE E
# Input: PLIP XML reports from Google Drive
# Output: Publication-quality fingerprint heatmaps + CSV
# ============================================================

!pip -q install pandas matplotlib seaborn

import os
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from google.colab import drive

# ============================================================
# 1. MOUNT GOOGLE DRIVE
# ============================================================

drive.mount("/content/drive")

# CHANGE THESE TWO PATHS
CALYCOSIN_XML = (
    "/content/drive/MyDrive/Plip/Calycosin/report.xml"
)

FORSYTHOSIDE_XML = (
    "/content/drive/MyDrive/Plip/Forsy/report.xml"
)

# Output folder
OUTPUT_DIR = "/content/drive/MyDrive/PLIF_Fingerprint_Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

for path in [CALYCOSIN_XML, FORSYTHOSIDE_XML]:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"\nFile not found:\n{path}\n"
            "Please correct the path to your XML report in Drive."
        )

print("Calycosin XML:", CALYCOSIN_XML)
print("Forsythoside E XML:", FORSYTHOSIDE_XML)
print("Output folder:", OUTPUT_DIR)

# ============================================================
# 2. INTERACTION CLASS DEFINITIONS
# ============================================================

INTERACTION_TAGS = {
    "hydrophobic_interactions": "Hydrophobic",
    "hydrogen_bonds": "H-bond",
    "salt_bridges": "Salt bridge",
    "water_bridges": "Water bridge",
    "pi_stacks": "Pi stacking",
    "pi_cation_interactions": "Pi-cation",
    "halogen_bonds": "Halogen bond",
    "metal_complexes": "Metal complex",
}

SHORT_LABELS = {
    "Hydrophobic": "Hy",
    "H-bond": "HD",
    "Salt bridge": "SB",
    "Water bridge": "WB",
    "Pi stacking": "Pi",
    "Pi-cation": "PC",
    "Halogen bond": "Ha",
    "Metal complex": "Me",
}

COLORS = {
    "None": "#FFFFFF",
    "Hydrophobic": "#F46D6D",
    "H-bond": "#F5C56B",
    "Salt bridge": "#36B98A",
    "Water bridge": "#62BCE8",
    "Pi stacking": "#A64CA6",
    "Pi-cation": "#A64CA6",
    "Halogen bond": "#8888D8",
    "Metal complex": "#777777",
    "Mixed": "#42A5A5",
}

# ============================================================
# 3. PARSE PLIP XML
# ============================================================

def parse_plip(xml_path, ligand_name):

    root = ET.parse(xml_path).getroot()
    records = []

    for site in root.findall(".//bindingsite"):

        smiles = site.findtext(
            "./identifiers/smiles", default=""
        )
        pdb_id = site.findtext("./pdbid", default="")

        # Include all binding sites found in the XML.
        interactions = site.find("./interactions")

        if interactions is None:
            continue

        for section in interactions:

            interaction_name = INTERACTION_TAGS.get(
                section.tag
            )

            if interaction_name is None:
                continue

            for interaction in list(section):

                residue_number = interaction.findtext(
                    "resnr"
                )
                residue_type = interaction.findtext(
                    "restype"
                )
                chain = interaction.findtext(
                    "reschain", default="A"
                )

                if (
                    residue_number is None
                    or residue_type is None
                ):
                    continue

                residue_number = int(residue_number)
                residue_type = residue_type.upper()

                # Chain is included to avoid merging residues
                # from different protein chains.
                residue = (
                    f"{residue_type}{residue_number}:{chain}"
                )

                distance = interaction.findtext("dist")

                records.append({
                    "Ligand": ligand_name,
                    "Residue": residue,
                    "ResidueNumber": residue_number,
                    "ResidueType": residue_type,
                    "Chain": chain,
                    "Interaction": interaction_name,
                    "Distance_A": distance,
                    "PDB_ID": pdb_id,
                    "SMILES": smiles,
                })

    df = pd.DataFrame(records)

    if df.empty:
        raise ValueError(
            f"No supported PLIP interactions found in {xml_path}"
        )

    df["Distance_A"] = pd.to_numeric(
        df["Distance_A"], errors="coerce"
    )

    return df


# ============================================================
# 4. READ BOTH REPORTS
# ============================================================

calycosin_df = parse_plip(
    CALYCOSIN_XML, "Calycosin"
)

forsythoside_df = parse_plip(
    FORSYTHOSIDE_XML, "Forsythoside E"
)

all_data = pd.concat(
    [calycosin_df, forsythoside_df],
    ignore_index=True
)

all_data.to_csv(
    os.path.join(OUTPUT_DIR, "PLIF_All_Interactions.csv"),
    index=False
)

print("\nInteraction counts:")
print(
    all_data.groupby(["Ligand", "Interaction"])
    .size()
    .unstack(fill_value=0)
    .to_string()
)

# ============================================================
# 5. CREATE A COMMON RESIDUE SET
# ============================================================

# Sort residues by numerical residue position.
residue_info = (
    all_data[
        ["Residue", "ResidueNumber", "Chain", "ResidueType"]
    ]
    .drop_duplicates()
    .sort_values(
        ["ResidueNumber", "Chain", "ResidueType"]
    )
)

residues = residue_info["Residue"].tolist()

ligands = ["Calycosin", "Forsythoside E"]

interaction_sets = {}

for ligand in ligands:

    subset = all_data[all_data["Ligand"] == ligand]

    interaction_sets[ligand] = {
        residue: sorted(set(group["Interaction"]))
        for residue, group in subset.groupby("Residue")
    }


# ============================================================
# 6. ENCODE INTERACTION FINGERPRINT
# ============================================================

# 0 = no interaction
# 1 = hydrophobic
# 2 = hydrogen bond
# 3 = salt bridge
# 4 = water bridge
# 5 = pi interaction
# 6 = halogen bond
# 7 = metal complex
# 8 = multiple interaction classes

PRIORITY = {
    "Hydrophobic": 1,
    "H-bond": 2,
    "Salt bridge": 3,
    "Water bridge": 4,
    "Pi stacking": 5,
    "Pi-cation": 5,
    "Halogen bond": 6,
    "Metal complex": 7,
}


def get_cell_code(interactions):

    if len(interactions) == 0:
        return 0

    # A separate mixed code is used if multiple distinct
    # interaction classes occur at the same residue.
    if len(interactions) > 1:
        return 8

    return PRIORITY.get(interactions[0], 8)


def get_cell_label(interactions):

    if len(interactions) == 0:
        return ""

    if len(interactions) == 1:
        return SHORT_LABELS.get(
            interactions[0], interactions[0][:2]
        )

    return "Mix"


matrix = pd.DataFrame(
    [
        [
            get_cell_code(
                interaction_sets[ligand].get(residue, [])
            )
            for residue in residues
        ]
        for ligand in ligands
    ],
    index=ligands,
    columns=residues
)

labels = pd.DataFrame(
    [
        [
            get_cell_label(
                interaction_sets[ligand].get(residue, [])
            )
            for residue in residues
        ]
        for ligand in ligands
    ],
    index=ligands,
    columns=residues
)

matrix.to_csv(
    os.path.join(
        OUTPUT_DIR, "PLIF_Fingerprint_Matrix.csv"
    )
)

labels.to_csv(
    os.path.join(
        OUTPUT_DIR, "PLIF_Fingerprint_Labels.csv"
    )
)

# ============================================================
# 7. CREATE PANELS A AND B
# ============================================================

CODE_COLORS = [
    COLORS["None"],
    COLORS["Hydrophobic"],
    COLORS["H-bond"],
    COLORS["Salt bridge"],
    COLORS["Water bridge"],
    COLORS["Pi stacking"],
    COLORS["Halogen bond"],
    COLORS["Metal complex"],
    COLORS["Mixed"],
]

cmap = ListedColormap(CODE_COLORS)

norm = BoundaryNorm(
    np.arange(-0.5, 9.5, 1),
    cmap.N
)

# Increase figure width to accommodate residue labels.
fig_width = max(14, len(residues) * 0.48)

fig, axes = plt.subplots(
    nrows=2,
    ncols=1,
    figsize=(fig_width, 5.8),
    gridspec_kw={"height_ratios": [1, 1]}
)

panel_details = [
    ("A", "Calycosin"),
    ("B", "Forsythoside E"),
]

for ax, (panel_letter, ligand) in zip(
    axes, panel_details
):

    sns.heatmap(
        matrix.loc[[ligand]],
        ax=ax,
        cmap=cmap,
        norm=norm,
        cbar=False,
        linewidths=0.8,
        linecolor="#FFFFFF",
        annot=labels.loc[[ligand]],
        fmt="",
        annot_kws={
            "fontsize": 8,
            "fontweight": "bold"
        }
    )

    ax.set_title(
        f"{panel_letter}. {ligand}",
        loc="left",
        fontsize=13,
        fontweight="bold",
        pad=8
    )

    ax.set_xlabel("")
    ax.set_ylabel("")

    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=90,
        fontsize=8
    )

    ax.set_yticklabels(
        [ligand],
        rotation=0,
        fontsize=9
    )

legend_items = [
    Patch(
        facecolor=COLORS["Hydrophobic"],
        label="Hydrophobic"
    ),
    Patch(
        facecolor=COLORS["H-bond"],
        label="H-bond"
    ),
    Patch(
        facecolor=COLORS["Salt bridge"],
        label="Salt bridge"
    ),
    Patch(
        facecolor=COLORS["Water bridge"],
        label="Water bridge"
    ),
    Patch(
        facecolor=COLORS["Pi stacking"],
        label="Pi interactions"
    ),
    Patch(
        facecolor=COLORS["Halogen bond"],
        label="Halogen bond"
    ),
    Patch(
        facecolor=COLORS["Metal complex"],
        label="Metal complex"
    ),
    Patch(
        facecolor=COLORS["Mixed"],
        label="Multiple classes"
    ),
]

fig.legend(
    handles=legend_items,
    loc="lower center",
    ncol=4,
    frameon=False,
    fontsize=8,
    bbox_to_anchor=(0.5, 0.015)
)

fig.suptitle(
    "Protein–Ligand Interaction Fingerprint",
    fontsize=15,
    fontweight="bold",
    y=0.98
)

plt.tight_layout(
    rect=[0, 0.12, 1, 0.94]
)

# ============================================================
# 8. SAVE FIGURES TO GOOGLE DRIVE
# ============================================================

png_path = os.path.join(
    OUTPUT_DIR,
    "PLIF_Calycosin_ForsythosideE.png"
)

pdf_path = os.path.join(
    OUTPUT_DIR,
    "PLIF_Calycosin_ForsythosideE.pdf"
)

fig.savefig(
    png_path,
    dpi=600,
    bbox_inches="tight",
    facecolor="white"
)

fig.savefig(
    pdf_path,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
plt.close(fig)

# ============================================================
# 9. SAVE A SIMPLE INTERACTION SUMMARY
# ============================================================

summary = (
    all_data.groupby(["Ligand", "Interaction"])
    .agg(
        Interaction_Records=("Residue", "size"),
        Unique_Residues=("Residue", "nunique")
    )
    .reset_index()
)

summary_path = os.path.join(
    OUTPUT_DIR,
    "PLIF_Interaction_Summary.csv"
)

summary.to_csv(summary_path, index=False)

print("\nSUCCESS — all results saved in Google Drive:")
for name in sorted(os.listdir(OUTPUT_DIR)):
    print(os.path.join(OUTPUT_DIR, name))
