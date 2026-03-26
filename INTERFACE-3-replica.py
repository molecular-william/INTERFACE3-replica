import sys
import os, glob
from io import StringIO
import numpy as np
import pandas as pd
from Bio import PDB
from collections import defaultdict, Counter
import py3Dmol
import random
from scipy.spatial import cKDTree
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QRadioButton,
    QDoubleSpinBox, QComboBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QGroupBox, QFileDialog, QMessageBox, QTableView,
    QListWidget, QListWidgetItem, QAbstractItemView, QProgressBar,
    QCheckBox, QSpinBox, QTabWidget
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QAbstractTableModel, QModelIndex
from PyQt6.QtWebEngineWidgets import QWebEngineView

# -------------------------------------------------------------------
# Van der Waals radii (in Å) for heavy atoms and H
# Values from Tsai et al. (1999) as used in the paper
# -------------------------------------------------------------------
VDW_RADII = {
        'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80, 'H': 1.20,
        'CA': 1.70, 'CB': 1.70, 'CD': 1.70, 'CG': 1.70, 'CE': 1.70,
        'NZ': 1.55, 'OD1': 1.52, 'OD2': 1.52, 'NE': 1.55, 'NH1': 1.55,
        'NH2': 1.55, 'OE1': 1.52, 'OE2': 1.52, 'ND2': 1.55, 'OG': 1.52,
        'OG1': 1.52, 'SG': 1.80, 'NE2': 1.55, 'ND1': 1.55, 'CE1': 1.70,
        'CD2': 1.70, 'CG1': 1.70, 'CG2': 1.70, 'CD1': 1.70, 'CE2': 1.70,
        'CZ': 1.70, 'OH': 1.52, 'NE1': 1.55, 'CZ2': 1.70, 'CZ3': 1.70,
        'CE3': 1.70, 'CH2': 1.70, 'SD': 1.80, 'CE': 1.70, 'DEFAULT': 1.70,
    }
# paper said increment by 0.5 to account for inaccuracies in atomic coords
for key in VDW_RADII.keys():
    VDW_RADII[key] += 0.25

def get_radius(element):
    """Return van der Waals radius for an element."""
    return VDW_RADII.get(element, VDW_RADII['DEFAULT'])

# -------------------------------------------------------------------
# Helper functions (most unchanged)
# -------------------------------------------------------------------
def _parse_helix_record(line: str) -> dict:
    data = line.split()
    helix_id = int(data[1])
    chain = data[4]
    start = int(data[5])
    end = int(data[8])
    return {'helix_id': helix_id, 'chain_id': chain,
            'start_residue': start, 'end_residue': end}

def parse_pdb_with_helix_from_string(pdb_string, atom_type='heavy'):
    # First pass: collect HELIX records
    helix_map = {}
    for line in pdb_string.splitlines():
        if line.startswith('HELIX'):
            rec = _parse_helix_record(line)
            for resnum in range(rec['start_residue'], rec['end_residue'] + 1):
                helix_map[(rec['chain_id'], resnum)] = rec['helix_id']

    # Second pass: parse structure
    handle = StringIO(pdb_string)
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', handle)

    coords = []
    atom_info = []

    for model in structure:
        for chain in model:
            for residue in chain:
                if not PDB.is_aa(residue):
                    continue
                key = (chain.id, residue.id[1])
                helix_id = helix_map.get(key, None)

                for atom in residue:
                    if atom_type == 'all' or (atom_type == 'heavy' and atom.element != 'H'):
                        coords.append(atom.coord)
                        atom_info.append({
                            'chain': chain.id,
                            'resnum': residue.id[1],
                            'resname': residue.resname,
                            'atom_name': atom.name,
                            'element': atom.element,
                            'helix_id': helix_id
                        })
        break  # only first model

    return np.array(coords), atom_info

def build_adjacency_from_radii(coords, atom_info, tolerance=0.5):
    """
    Build adjacency list based on van der Waals contact.
    Returns list of sets (adjacency per atom index).
    """
    n_atoms = len(coords)
    radii = np.array([get_radius(info['element']) for info in atom_info])
    # Use a k‑d tree to quickly find candidate pairs within a loose cutoff
    # (max possible distance = sum of two radii + tolerance)
    max_rad = np.max(radii)
    max_cutoff = 2 * max_rad + tolerance
    tree = cKDTree(coords)
    # We'll collect edges in a set to avoid duplicates
    edges = set()
    for i in range(n_atoms):
        # query all atoms within max_cutoff
        indices = tree.query_ball_point(coords[i], max_cutoff)
        for j in indices:
            if j <= i:
                continue
            d = np.linalg.norm(coords[i] - coords[j])
            if d <= radii[i] + radii[j] + tolerance:
                edges.add((i, j))
    # Build adjacency
    adj = [set() for _ in range(n_atoms)]
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
    return adj

def find_triangles(adj):
    triangles = []
    n = len(adj)
    for i in range(n):
        neighbors_i = adj[i]
        for j in neighbors_i:
            if j <= i:
                continue
            common = neighbors_i.intersection(adj[j])
            for k in common:
                if k > j:
                    triangles.append((i, j, k))
    return triangles

def one_letter_code(resname):
    three_to_one = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
        'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
    }
    return three_to_one.get(resname.upper(), 'X')

def filter_triangles(triangles, atom_info):
    valid = []
    for (i, j, k) in triangles:
        res_keys = (
            (atom_info[i]['chain'], atom_info[i]['resnum']),
            (atom_info[j]['chain'], atom_info[j]['resnum']),
            (atom_info[k]['chain'], atom_info[k]['resnum'])
        )
        if len(set(res_keys)) < 3:
            continue

        hids = {atom_info[x]['helix_id'] for x in (i, j, k)}
        if None in hids or len(hids) < 2:
            continue

        codes = [one_letter_code(atom_info[x]['resname']) for x in (i, j, k)]
        triplet_type = ''.join(sorted(codes))

        valid.append({
            'atoms': (i, j, k),
            'triplet_type': triplet_type
        })
    return valid

def get_residue_sphere_coords_from_set(atom_info, coords, residue_set):
    atom_spheres = []
    label_candidates = {}

    for idx, info in enumerate(atom_info):
        key = (info['chain'], info['resnum'])
        if key in residue_set:
            atom_spheres.append((
                info['chain'],
                info['resnum'],
                info['resname'],
                info['atom_name'],
                info['element'],
                coords[idx][0],
                coords[idx][1],
                coords[idx][2]
            ))

            if key not in label_candidates:
                label_candidates[key] = (info['resname'], coords[idx], info['atom_name'])
            else:
                if info['atom_name'] == 'CA' and label_candidates[key][2] != 'CA':
                    label_candidates[key] = (info['resname'], coords[idx], info['atom_name'])

    label_positions = []
    for (chain, resnum), (resname, coord, _) in label_candidates.items():
        label_positions.append((chain, resnum, resname, coord[0], coord[1], coord[2]))

    return atom_spheres, label_positions

def render_py3dmol(pdb_string, atom_spheres, label_positions):
    color_map = {
        'C': 'gray', 'N': 'blue', 'O': 'red', 'S': 'yellow',
        'H': 'white', 'CA': 'green', 'CB': 'gray', 'CD': 'gray',
        'CG': 'gray', 'CE': 'gray', 'NZ': 'blue', 'OD1': 'red',
        'OD2': 'red', 'NE': 'blue', 'NH1': 'blue', 'NH2': 'blue',
        'OE1': 'red', 'OE2': 'red', 'ND2': 'blue', 'OG': 'red',
        'OG1': 'red', 'SG': 'yellow', 'NE2': 'blue', 'ND1': 'blue',
        'CE1': 'gray', 'CD2': 'gray', 'CG1': 'gray', 'CG2': 'gray',
        'CD1': 'gray', 'CE2': 'gray', 'CZ': 'gray', 'OH': 'red',
        'NE1': 'blue', 'CZ2': 'gray', 'CZ3': 'gray', 'CE3': 'gray',
        'CH2': 'gray', 'SD': 'yellow', 'DEFAULT': 'gray',
    }

    radius_map = {
        'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80, 'H': 1.20,
        'CA': 1.70, 'CB': 1.70, 'CD': 1.70, 'CG': 1.70, 'CE': 1.70,
        'NZ': 1.55, 'OD1': 1.52, 'OD2': 1.52, 'NE': 1.55, 'NH1': 1.55,
        'NH2': 1.55, 'OE1': 1.52, 'OE2': 1.52, 'ND2': 1.55, 'OG': 1.52,
        'OG1': 1.52, 'SG': 1.80, 'NE2': 1.55, 'ND1': 1.55, 'CE1': 1.70,
        'CD2': 1.70, 'CG1': 1.70, 'CG2': 1.70, 'CD1': 1.70, 'CE2': 1.70,
        'CZ': 1.70, 'OH': 1.52, 'NE1': 1.55, 'CZ2': 1.70, 'CZ3': 1.70,
        'CE3': 1.70, 'CH2': 1.70, 'SD': 1.80, 'CE': 1.70, 'DEFAULT': 1.70,
    }

    view = py3Dmol.view(width=800, height=600)
    view.addModel(pdb_string, 'pdb')
    view.setStyle({'cartoon': {'color': 'spectrum'}})

    for (chain, resnum, resname, atom_name, element, x, y, z) in atom_spheres:
        colour = color_map.get(element, color_map['DEFAULT'])
        radius = radius_map.get(element, radius_map['DEFAULT'])
        view.addSphere({
            'center': {'x': float(x), 'y': float(y), 'z': float(z)},
            'radius': radius,
            'color': colour,
            'alpha': 0.98
        })

    for (chain, resnum, resname, x, y, z) in label_positions:
        view.addLabel(f"{chain}:{resnum} {resname}", {
            'position': {'x': float(x), 'y': float(y), 'z': float(z)},
            'fontSize': 12,
        })

    view.zoomTo()
    return view._make_html()

# -------------------------------------------------------------------
# Propensity calculation (unchanged)
# -------------------------------------------------------------------
HEAVY_ATOM_COUNTS = {
    'A': 5, 'R': 11, 'N': 8, 'D': 8, 'C': 6, 'Q': 9, 'E': 9, 'G': 4,
    'H': 10, 'I': 8, 'L': 8, 'K': 9, 'M': 8, 'F': 11, 'P': 7, 'S': 6,
    'T': 7, 'W': 14, 'Y': 12, 'V': 7
}

class PropensityCalculator:
    def __init__(self, file_data_list, bootstrap_samples=0, random_seed=42):
        self.file_data = file_data_list
        self.bootstrap_samples = bootstrap_samples
        random.seed(random_seed)

    def compute(self):
        total_trip_counts = Counter()
        total_res_counts = Counter()
        for trip_counts, res_counts in self.file_data:
            total_trip_counts.update(trip_counts)
            total_res_counts.update(res_counts)

        total_atom_triplets = sum(total_trip_counts.values())
        n_total_atoms = sum(total_res_counts[res] * HEAVY_ATOM_COUNTS.get(res, 0) for res in total_res_counts)

        results = {}
        for tt, obs_count in total_trip_counts.items():
            obs_prob = obs_count / total_atom_triplets
            exp_prob = self._expected_prob(tt, total_res_counts, n_total_atoms)
            propensity = obs_prob / exp_prob if exp_prob > 0 else np.nan
            results[tt] = {
                'obs_count': obs_count,
                'exp_prob': exp_prob,
                'propensity': propensity
            }

        if self.bootstrap_samples > 0:
            boot_props = {tt: [] for tt in results}
            n_files = len(self.file_data)
            file_trip_counts = [tc for tc, _ in self.file_data]
            file_res_counts = [rc for _, rc in self.file_data]

            for _ in range(self.bootstrap_samples):
                idxs = random.choices(range(n_files), k=n_files)
                boot_trip = Counter()
                boot_res = Counter()
                for i in idxs:
                    boot_trip.update(file_trip_counts[i])
                    boot_res.update(file_res_counts[i])

                boot_total_triplets = sum(boot_trip.values())
                boot_n_atoms = sum(boot_res[res] * HEAVY_ATOM_COUNTS.get(res, 0) for res in boot_res)
                if boot_total_triplets == 0:
                    continue

                for tt in results:
                    obs = boot_trip[tt] / boot_total_triplets
                    exp = self._expected_prob(tt, boot_res, boot_n_atoms)
                    if exp > 0:
                        boot_props[tt].append(obs / exp)
                    else:
                        boot_props[tt].append(np.nan)

            for tt in results:
                props = [p for p in boot_props[tt] if not np.isnan(p)]
                if len(props) >= 2:
                    results[tt]['ci_lower'] = np.percentile(props, 2.5)
                    results[tt]['ci_upper'] = np.percentile(props, 97.5)
                else:
                    results[tt]['ci_lower'] = results[tt]['ci_upper'] = np.nan

        return results

    @staticmethod
    def _expected_prob(triplet_type, residue_counts, n_total_atoms):
        types = sorted(triplet_type)
        N = {res: residue_counts.get(res, 0) for res in types}
        n = {res: HEAVY_ATOM_COUNTS.get(res, 0) for res in types}
        if types[0] == types[1] == types[2]:
            mult = 1
        elif types[0] == types[1] or types[1] == types[2]:
            mult = 3
        else:
            mult = 6
        prod = 1.0
        for res in types:
            prod *= N[res] * n[res]
        denom = (n_total_atoms ** 3) / mult
        return prod / denom if denom != 0 else 0.0

# -------------------------------------------------------------------
# Worker for geometry computation (now using radii contact)
# -------------------------------------------------------------------
class ComputeWorker(QThread):
    finished = pyqtSignal(str, object, object, object, object, object)
    error = pyqtSignal(str)

    def __init__(self, filename, pdb_bytes, atom_type, alpha, tolerance=0.5):
        super().__init__()
        self.filename = filename
        self.pdb_bytes = pdb_bytes
        self.atom_type = atom_type
        self.alpha = alpha          # Not used in new method, kept for compatibility
        self.tolerance = tolerance

    def run(self):
        try:
            pdb_string = self.pdb_bytes.decode('utf-8')
            coords, atom_info = parse_pdb_with_helix_from_string(pdb_string, self.atom_type)

            if len(coords) < 3:
                self.error.emit(f"{self.filename}: Need at least 3 atoms to form triangles.")
                return

            # Build adjacency based on van der Waals radii + tolerance
            adj = build_adjacency_from_radii(coords, atom_info, self.tolerance)
            triangles = find_triangles(adj)
            valid_triplets = filter_triangles(triangles, atom_info)
            type_counts = Counter(t['triplet_type'] for t in valid_triplets)

            self.finished.emit(self.filename, pdb_string, atom_info, coords, valid_triplets, type_counts)

        except Exception as e:
            self.error.emit(f"{self.filename}: {str(e)}")

# -------------------------------------------------------------------
# Propensity worker (unchanged)
# -------------------------------------------------------------------
class PropensityWorker(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, file_data_list, bootstrap_samples):
        super().__init__()
        self.file_data_list = file_data_list
        self.bootstrap_samples = bootstrap_samples

    def run(self):
        try:
            calc = PropensityCalculator(self.file_data_list, self.bootstrap_samples)
            results = calc.compute()
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))

# -------------------------------------------------------------------
# Triplet Table Model (unchanged)
# -------------------------------------------------------------------
class TripletTableModel(QAbstractTableModel):
    def __init__(self):
        super().__init__()
        self._data = pd.DataFrame(columns=["Triplet Type", "Count"])

    def rowCount(self, parent=QModelIndex()):
        return len(self._data)

    def columnCount(self, parent=QModelIndex()):
        return len(self._data.columns)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            value = self._data.iloc[index.row(), index.column()]
            return str(value)
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return self._data.columns[section]
            else:
                return str(section + 1)
        return None

    def sort(self, column, order=Qt.SortOrder.AscendingOrder):
        self.layoutAboutToBeChanged.emit()
        self._data.sort_values(by=self._data.columns[column],
                               ascending=order == Qt.SortOrder.AscendingOrder,
                               inplace=True)
        self._data.reset_index(drop=True, inplace=True)
        self.layoutChanged.emit()

    def update_data(self, type_counts):
        self.layoutAboutToBeChanged.emit()
        df = pd.DataFrame(list(type_counts.items()), columns=["Triplet Type", "Count"])
        df.sort_values(by="Triplet Type", inplace=True, ascending=False)
        df.reset_index(drop=True, inplace=True)
        self._data = df
        self.layoutChanged.emit()

    def clear(self):
        self.layoutAboutToBeChanged.emit()
        self._data = pd.DataFrame(columns=["Triplet Type", "Count"])
        self.layoutChanged.emit()

# -------------------------------------------------------------------
# Main Window (with tabs, propensity moved to right)
# -------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("INTERFACE3-replica (with van der Waals contacts)")
        self.setMinimumSize(1300, 800)

        self.file_results = {}
        self.current_filename = None

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # Left panel (controls + triplet table)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Directory selection
        dir_group = QGroupBox("PDB Directory")
        dir_layout = QVBoxLayout()
        self.dir_path_edit = QLineEdit()
        self.dir_path_edit.setReadOnly(True)
        self.browse_dir_btn = QPushButton("Browse Directory")
        self.browse_dir_btn.clicked.connect(self.browse_directory)
        dir_layout.addWidget(self.dir_path_edit)
        dir_layout.addWidget(self.browse_dir_btn)
        dir_group.setLayout(dir_layout)
        left_layout.addWidget(dir_group)

        # File list
        file_list_group = QGroupBox("PDB Files")
        file_list_layout = QVBoxLayout()
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        file_list_layout.addWidget(self.file_list)
        file_list_group.setLayout(file_list_layout)
        left_layout.addWidget(file_list_group)

        # Compute buttons
        compute_btns_group = QGroupBox("Compute")
        compute_btns_layout = QHBoxLayout()
        self.compute_selected_btn = QPushButton("Compute Selected")
        self.compute_selected_btn.clicked.connect(self.compute_selected)
        self.compute_all_btn = QPushButton("Compute All")
        self.compute_all_btn.clicked.connect(self.compute_all)
        compute_btns_layout.addWidget(self.compute_selected_btn)
        compute_btns_layout.addWidget(self.compute_all_btn)
        compute_btns_group.setLayout(compute_btns_layout)
        left_layout.addWidget(compute_btns_group)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)

        # Visualization file selection
        vis_group = QGroupBox("Visualize File")
        vis_layout = QVBoxLayout()
        self.vis_file_combo = QComboBox()
        self.vis_file_combo.currentTextChanged.connect(self.on_vis_file_changed)
        vis_layout.addWidget(self.vis_file_combo)
        vis_group.setLayout(vis_layout)
        left_layout.addWidget(vis_group)

        # Atom type selection
        atom_group = QGroupBox("Atom Selection")
        atom_layout = QVBoxLayout()
        self.heavy_radio = QRadioButton("heavy (exclude H)")
        self.heavy_radio.setChecked(True)
        self.all_radio = QRadioButton("all (include H)")
        atom_layout.addWidget(self.heavy_radio)
        atom_layout.addWidget(self.all_radio)
        atom_group.setLayout(atom_layout)
        left_layout.addWidget(atom_group)

        # Info label
        self.info_label = QLabel("No data")
        left_layout.addWidget(self.info_label)

        # Triplet type counts table
        table_group = QGroupBox("Triplet Type Counts")
        table_layout = QVBoxLayout()
        self.type_table = QTableView()
        self.type_table.horizontalHeader().setStretchLastSection(True)
        self.type_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table_model = TripletTableModel()
        self.type_table.setModel(self.table_model)
        table_layout.addWidget(self.type_table)
        table_group.setLayout(table_layout)
        left_layout.addWidget(table_group)

        # Dropdown for selecting type to visualise
        select_layout = QHBoxLayout()
        select_layout.addWidget(QLabel("Show type:"))
        self.type_combo = QComboBox()
        self.type_combo.currentTextChanged.connect(self.on_type_selected)
        self.type_combo.setMaxVisibleItems(10)
        select_layout.addWidget(self.type_combo)
        left_layout.addLayout(select_layout)

        # Right panel: TabWidget with 3D view and propensity analysis
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        self.tab_widget = QTabWidget()
        right_layout.addWidget(self.tab_widget)

        # Tab 1: 3D View
        self.web_view = QWebEngineView()
        self.tab_widget.addTab(self.web_view, "3D View")

        # Tab 2: Propensity Analysis
        prop_widget = QWidget()
        prop_layout = QVBoxLayout(prop_widget)

        self.use_all_files_cb = QCheckBox("Use all loaded files")
        self.use_all_files_cb.setChecked(True)
        prop_layout.addWidget(self.use_all_files_cb)

        self.bootstrap_spin = QSpinBox()
        self.bootstrap_spin.setRange(0, 10000)
        self.bootstrap_spin.setValue(0)
        self.bootstrap_spin.setSuffix(" samples")
        prop_layout.addWidget(QLabel("Bootstrap samples:"))
        prop_layout.addWidget(self.bootstrap_spin)

        self.compute_prop_btn = QPushButton("Compute Propensity")
        self.compute_prop_btn.clicked.connect(self.compute_propensity)
        prop_layout.addWidget(self.compute_prop_btn)

        self.prop_progress = QProgressBar()
        self.prop_progress.setVisible(False)
        prop_layout.addWidget(self.prop_progress)

        self.prop_table = QTableWidget()
        self.prop_table.setColumnCount(6)
        self.prop_table.setHorizontalHeaderLabels(["Triplet", "Observed", "Expected Prob", "Propensity", "CI Lower", "CI Upper"])
        self.prop_table.horizontalHeader().setStretchLastSection(True)
        prop_layout.addWidget(self.prop_table)

        self.tab_widget.addTab(prop_widget, "Propensity Analysis")

        main_layout.addWidget(left_panel, 1)
        main_layout.addWidget(right_panel, 2)

        self.worker = None
        self.compute_queue = []
        self.computing_all = False
        self.update_ui_after_compute()

    # -------------------------------------------------------------------
    # Directory and file handling
    # -------------------------------------------------------------------
    def browse_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory with PDB files")
        if dir_path:
            self.dir_path_edit.setText(dir_path)
            self.populate_file_list(dir_path)

    def populate_file_list(self, dir_path):
        self.file_list.clear()
        pdb_files = glob.glob(f"{dir_path}/*.pdb")
        for fname in pdb_files:
            self.file_list.addItem(os.path.basename(fname))
        if not pdb_files:
            QMessageBox.information(self, "No files", "No .pdb files found in the selected directory.")

    # -------------------------------------------------------------------
    # Computation
    # -------------------------------------------------------------------
    def compute_selected(self):
        current_item = self.file_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "No selection", "Please select a file from the list.")
            return
        filename = current_item.text()
        full_path = os.path.join(self.dir_path_edit.text(), filename)
        self._start_computation(full_path)

    def compute_all(self):
        if self.computing_all:
            return
        self.compute_queue = []
        for i in range(self.file_list.count()):
            filename = self.file_list.item(i).text()
            full_path = os.path.join(self.dir_path_edit.text(), filename)
            self.compute_queue.append(full_path)
        if not self.compute_queue:
            QMessageBox.warning(self, "No files", "No PDB files to compute.")
            return
        self.computing_all = True
        self.compute_total = len(self.compute_queue)
        self.compute_processed = 0
        self.progress_bar.setRange(0, self.compute_total)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self._process_next_compute()

    def _process_next_compute(self):
        if not self.compute_queue:
            self.computing_all = False
            self._enable_ui(True)
            self.progress_bar.setVisible(False)
            QMessageBox.information(self, "Complete", f"Computed {len(self.file_results)} file(s).")
            return
        full_path = self.compute_queue.pop(0)
        self._start_computation(full_path, is_all=True)

    def _start_computation(self, full_path, is_all=False):
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.warning(self, "Busy", "A computation is already running. Please wait.")
            return

        try:
            with open(full_path, 'rb') as f:
                pdb_bytes = f.read()
        except Exception as e:
            QMessageBox.critical(self, "Read error", f"Cannot read {full_path}: {e}")
            if is_all:
                self._process_next_compute()
            return

        atom_type = 'heavy' if self.heavy_radio.isChecked() else 'all'
        alpha = self.alpha_spin.value()   # not used but kept

        self._disable_ui()

        self.worker = ComputeWorker(full_path, pdb_bytes, atom_type, alpha)
        self.worker.finished.connect(self.on_computation_finished)
        self.worker.error.connect(self.on_computation_error)
        self.worker.finished.connect(lambda: self._on_worker_finished(is_all, full_path))
        self.worker.start()

    def _on_worker_finished(self, is_all, filename):
        self.worker = None
        if is_all:
            self.compute_processed += 1
            self.progress_bar.setValue(self.compute_processed)
            self._process_next_compute()
        else:
            self._enable_ui(True)

    def _disable_ui(self):
        self.compute_selected_btn.setEnabled(False)
        self.compute_all_btn.setEnabled(False)
        self.browse_dir_btn.setEnabled(False)
        self.file_list.setEnabled(False)
        self.vis_file_combo.setEnabled(False)
        self.type_combo.setEnabled(False)
        self.info_label.setText("Computing...")

    def _enable_ui(self, enable):
        self.compute_selected_btn.setEnabled(enable)
        self.compute_all_btn.setEnabled(enable)
        self.browse_dir_btn.setEnabled(enable)
        self.file_list.setEnabled(enable)
        self.vis_file_combo.setEnabled(enable)
        self.type_combo.setEnabled(enable)
        if not enable:
            self.info_label.setText("Computing...")
        else:
            self.update_ui_after_compute()

    def on_computation_finished(self, filename, pdb_string, atom_info, coords, valid_triplets, type_counts):
        # Build type_to_residues
        type_to_residues = {}
        for trip in valid_triplets:
            typ = trip['triplet_type']
            if typ not in type_to_residues:
                type_to_residues[typ] = set()
            i, j, k = trip['atoms']
            type_to_residues[typ].add((atom_info[i]['chain'], atom_info[i]['resnum']))
            type_to_residues[typ].add((atom_info[j]['chain'], atom_info[j]['resnum']))
            type_to_residues[typ].add((atom_info[k]['chain'], atom_info[k]['resnum']))

        # Compute residue_counts
        all_residues = set()
        for res_set in type_to_residues.values():
            all_residues.update(res_set)
        residue_counts = Counter()
        resname_lookup = {}
        for info in atom_info:
            key = (info['chain'], info['resnum'])
            if key not in resname_lookup:
                resname_lookup[key] = one_letter_code(info['resname'])
        for key in all_residues:
            resname = resname_lookup.get(key)
            if resname:
                residue_counts[resname] += 1

        self.file_results[filename] = {
            'pdb_string': pdb_string,
            'atom_info': atom_info,
            'coords': coords,
            'valid_triplets': valid_triplets,
            'type_counts': type_counts,
            'type_to_residues': type_to_residues,
            'residue_counts': residue_counts,
            'html_cache': {}
        }

        display_name = os.path.basename(filename)
        if self.vis_file_combo.findText(display_name) == -1:
            self.vis_file_combo.addItem(display_name, filename)
            if self.vis_file_combo.count() == 1:
                self.vis_file_combo.setCurrentIndex(0)
                self.current_filename = filename
                self._update_ui_for_file(filename)
        else:
            if self.current_filename == filename:
                self._update_ui_for_file(filename)

        if self.current_filename == filename:
            self.update_ui_after_compute()

    def on_computation_error(self, error_msg):
        QMessageBox.critical(self, "Computation Error", error_msg)
        if self.computing_all:
            self.compute_processed += 1
            self.progress_bar.setValue(self.compute_processed)
            self._process_next_compute()

    def update_ui_after_compute(self):
        if self.current_filename is None:
            self.info_label.setText("No file selected.")
            self.table_model.clear()
            self.type_combo.clear()
            self.web_view.setHtml("")
            return
        data = self.file_results.get(self.current_filename)
        if data is None:
            self.info_label.setText("No data for selected file.")
            self.table_model.clear()
            self.type_combo.clear()
            self.web_view.setHtml("")
            return

        n_atoms = len(data['coords'])
        n_triplets = len(data['valid_triplets'])
        self.info_label.setText(f"Atoms selected: {n_atoms}   |   Valid triplets: {n_triplets}")

        self.table_model.update_data(data['type_counts'])

        types = list(data['type_counts'].keys())
        self.type_combo.clear()
        if types:
            self.type_combo.addItems(sorted(types))
            self.type_combo.setEnabled(True)
            self.on_type_selected(types[0])
        else:
            self.type_combo.setEnabled(False)
            self.web_view.setHtml("<p>No triplets found for this file.</p>")

    def _update_ui_for_file(self, filename):
        self.current_filename = filename
        self.update_ui_after_compute()

    def on_vis_file_changed(self, display_name):
        idx = self.vis_file_combo.currentIndex()
        if idx >= 0:
            full_path = self.vis_file_combo.itemData(idx)
            if full_path and full_path in self.file_results:
                self._update_ui_for_file(full_path)

    def on_type_selected(self, triplet_type):
        if not triplet_type or self.current_filename is None:
            return
        data = self.file_results.get(self.current_filename)
        if data is None:
            return

        html_cache = data['html_cache']
        if triplet_type in html_cache:
            html = html_cache[triplet_type]
        else:
            residue_set = data['type_to_residues'].get(triplet_type, set())
            if not residue_set:
                return
            atom_spheres, label_positions = get_residue_sphere_coords_from_set(
                data['atom_info'], data['coords'], residue_set
            )
            html = render_py3dmol(data['pdb_string'], atom_spheres, label_positions)
            html_cache[triplet_type] = html
        self.web_view.setHtml(html)

    # -------------------------------------------------------------------
    # Propensity calculation
    # -------------------------------------------------------------------
    def compute_propensity(self):
        if not self.file_results:
            QMessageBox.warning(self, "No data", "No PDB files have been computed yet.")
            return
        if self.use_all_files_cb.isChecked():
            files = list(self.file_results.keys())
        else:
            files = list(self.file_results.keys())   # could be extended with selection
        if not files:
            QMessageBox.warning(self, "No files", "No files selected for propensity analysis.")
            return

        file_data_list = []
        for fname in files:
            data = self.file_results[fname]
            file_data_list.append((data['type_counts'], data['residue_counts']))

        bootstrap_samples = self.bootstrap_spin.value()
        self.prop_worker = PropensityWorker(file_data_list, bootstrap_samples)
        self.prop_worker.finished.connect(self.on_propensity_finished)
        self.prop_worker.error.connect(self.on_propensity_error)
        self.prop_worker.start()
        self.compute_prop_btn.setEnabled(False)
        self.prop_progress.setVisible(True)
        self.prop_progress.setRange(0, 0)

    def on_propensity_finished(self, results):
        self.compute_prop_btn.setEnabled(True)
        self.prop_progress.setVisible(False)
        self.prop_table.setRowCount(len(results))
        for i, (tt, info) in enumerate(sorted(results.items())):
            self.prop_table.setItem(i, 0, QTableWidgetItem(tt))
            self.prop_table.setItem(i, 1, QTableWidgetItem(str(info['obs_count'])))
            self.prop_table.setItem(i, 2, QTableWidgetItem(f"{info['exp_prob']:.5g}"))
            self.prop_table.setItem(i, 3, QTableWidgetItem(f"{info['propensity']:.3f}"))
            if 'ci_lower' in info:
                self.prop_table.setItem(i, 4, QTableWidgetItem(f"{info['ci_lower']:.3f}"))
                self.prop_table.setItem(i, 5, QTableWidgetItem(f"{info['ci_upper']:.3f}"))
            else:
                self.prop_table.setItem(i, 4, QTableWidgetItem(""))
                self.prop_table.setItem(i, 5, QTableWidgetItem(""))
        self.prop_table.resizeColumnsToContents()

    def on_propensity_error(self, err):
        self.compute_prop_btn.setEnabled(True)
        self.prop_progress.setVisible(False)
        QMessageBox.critical(self, "Propensity error", err)

# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('fusion')
    app.setStyleSheet('QComboBox {combobox-popup: 0}')
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
