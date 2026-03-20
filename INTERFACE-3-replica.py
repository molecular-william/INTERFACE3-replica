import sys
import os
from io import StringIO
import numpy as np
import pandas as pd
from Bio import PDB
import alphashape
from collections import defaultdict, Counter
import py3Dmol

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QRadioButton, QButtonGroup,
    QDoubleSpinBox, QComboBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QGroupBox, QFileDialog, QMessageBox, QTableView
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QAbstractTableModel, QModelIndex
from PyQt6.QtWebEngineWidgets import QWebEngineView


# -------------------------------------------------------------------
# Helper functions (identical to the original code)
# -------------------------------------------------------------------
def _parse_helix_record(line: str) -> dict:
    data = line.split()
    helix_id = int(data[2])
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

    # Second pass: parse structure from the same string
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


def compute_alpha_edges(coords, alpha):
    simplices_data = list(alphashape.alphasimplices(coords))
    valid_simplices = [s for s, r in simplices_data if r <= alpha]

    edges = set()
    for simplex in valid_simplices:
        for i in range(4):
            for j in range(i + 1, 4):
                a, b = simplex[i], simplex[j]
                edges.add((min(a, b), max(a, b)))
    return edges


def build_adjacency(edges, n_atoms):
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
    """
    Given atom_info, coords, and a set of (chain, resnum) tuples,
    return atom_spheres and label_positions for those residues.
    """
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
# Worker thread
# -------------------------------------------------------------------
class ComputeWorker(QThread):
    finished = pyqtSignal(object, object, object, object, object)
    error = pyqtSignal(str)

    def __init__(self, pdb_bytes, atom_type, alpha):
        super().__init__()
        self.pdb_bytes = pdb_bytes
        self.atom_type = atom_type
        self.alpha = alpha

    def run(self):
        try:
            pdb_string = self.pdb_bytes.decode('utf-8')
            coords, atom_info = parse_pdb_with_helix_from_string(pdb_string, self.atom_type)

            if len(coords) < 4:
                self.error.emit("Need at least 4 atoms. Cannot compute alpha shape.")
                return

            edges = compute_alpha_edges(coords, self.alpha)
            adj = build_adjacency(edges, len(coords))
            triangles = find_triangles(adj)
            valid_triplets = filter_triangles(triangles, atom_info)
            type_counts = Counter(t['triplet_type'] for t in valid_triplets)

            self.finished.emit(pdb_string, atom_info, coords, valid_triplets, type_counts)

        except Exception as e:
            self.error.emit(str(e))

# -------------------------------------------------------------------
# Pandas Table Model
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
                return str(section + 1)   # row numbers
        return None

    def sort(self, column, order=Qt.SortOrder.AscendingOrder):
        self.layoutAboutToBeChanged.emit()
        self._data.sort_values(by=self._data.columns[column],
                               ascending=order == Qt.SortOrder.AscendingOrder,
                               inplace=True)
        self._data.reset_index(drop=True, inplace=True)
        self.layoutChanged.emit()

    def update_data(self, type_counts):
        """Replace the model's data with a new dictionary of counts."""
        self.layoutAboutToBeChanged.emit()
        df = pd.DataFrame(list(type_counts.items()), columns=["Triplet Type", "Count"])
        df.sort_values(by="Count", inplace=True, ascending=False)   # optional initial sort
        df.reset_index(drop=True, inplace=True)
        self._data = df
        self.layoutChanged.emit()

    def clear(self):
        """Remove all data from the model."""
        self.layoutAboutToBeChanged.emit()
        self._data = pd.DataFrame(columns=["Triplet Type", "Count"])
        self.layoutChanged.emit()

# -------------------------------------------------------------------
# Main Window
# -------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("INTERFACE3-replica")
        self.setMinimumSize(1200, 800)

        self.pdb_bytes = None
        self.pdb_string = None
        self.atom_info = None
        self.coords = None
        self.valid_triplets = []
        self.type_counts = {}
        self.type_to_residues = {}
        self.html_cache = {}

        # Central widget and main horizontal layout
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # Left panel (controls + table)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # File selection
        file_group = QGroupBox("PDB File")
        file_layout = QVBoxLayout()
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setReadOnly(True)
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.clicked.connect(self.browse_file)
        file_layout.addWidget(self.file_path_edit)
        file_layout.addWidget(self.browse_btn)
        file_group.setLayout(file_layout)
        left_layout.addWidget(file_group)

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

        # Alpha value
        alpha_group = QGroupBox("Alpha (Å)")
        alpha_layout = QVBoxLayout()
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(1.0, 8.0)
        self.alpha_spin.setSingleStep(0.2)
        self.alpha_spin.setValue(3.0)
        alpha_layout.addWidget(self.alpha_spin)
        alpha_group.setLayout(alpha_layout)
        left_layout.addWidget(alpha_group)

        # Compute button
        self.compute_btn = QPushButton("Compute Triplets")
        self.compute_btn.clicked.connect(self.start_computation)
        left_layout.addWidget(self.compute_btn)

        # Info label (atoms, triplets)
        self.info_label = QLabel("No data")
        left_layout.addWidget(self.info_label)

        # Table of triplet type counts
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

        # Right panel (web view)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        self.web_view = QWebEngineView()
        right_layout.addWidget(self.web_view)

        # Add panels to main layout
        main_layout.addWidget(left_panel, 1)   # left takes 1 part
        main_layout.addWidget(right_panel, 3)  # right takes 3 parts

        self.worker = None
        self.update_ui_after_compute()

    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open PDB file", "", "PDB files (*.pdb)")
        if file_path:
            self.file_path_edit.setText(file_path)
            with open(file_path, 'rb') as f:
                self.pdb_bytes = f.read()

    def start_computation(self):
        if self.pdb_bytes is None:
            QMessageBox.warning(self, "No file", "Please select a PDB file first.")
            return

        atom_type = 'heavy' if self.heavy_radio.isChecked() else 'all'
        alpha = self.alpha_spin.value()

        self.compute_btn.setEnabled(False)
        self.info_label.setText("Computing... Please wait.")
        self.web_view.setHtml("")
        self.table_model.clear()
        self.type_combo.clear()

        self.worker = ComputeWorker(self.pdb_bytes, atom_type, alpha)
        self.worker.finished.connect(self.on_computation_finished)
        self.worker.error.connect(self.on_computation_error)
        self.worker.start()

    def on_computation_finished(self, pdb_string, atom_info, coords, valid_triplets, type_counts):
        self.pdb_string = pdb_string
        self.atom_info = atom_info
        self.coords = coords
        self.valid_triplets = valid_triplets
        self.type_counts = type_counts

        # --- New code: precompute residue sets per type ---
        self.type_to_residues = {}
        for trip in valid_triplets:
            typ = trip['triplet_type']
            if typ not in self.type_to_residues:
                self.type_to_residues[typ] = set()
            i, j, k = trip['atoms']
            self.type_to_residues[typ].add((atom_info[i]['chain'], atom_info[i]['resnum']))
            self.type_to_residues[typ].add((atom_info[j]['chain'], atom_info[j]['resnum']))
            self.type_to_residues[typ].add((atom_info[k]['chain'], atom_info[k]['resnum']))
        # ---------------------------------------------------
        self.table_model.update_data(type_counts)
        self.html_cache.clear()
        self.update_ui_after_compute()
        self.compute_btn.setEnabled(True)

    def on_computation_error(self, error_msg):
        QMessageBox.critical(self, "Computation Error", error_msg)
        self.info_label.setText("Error: " + error_msg)
        self.compute_btn.setEnabled(True)

    def update_ui_after_compute(self):
        if not self.valid_triplets:
            self.info_label.setText("No valid triplets found.")
            self.table_model.clear()
            self.type_combo.clear()
            self.type_combo.setMaxVisibleItems(10)
            self.type_combo.setEnabled(False)
            return

        n_atoms = len(self.coords)
        n_triplets = len(self.valid_triplets)
        self.info_label.setText(f"Atoms selected: {n_atoms}   |   Valid triplets: {n_triplets}")

        # Fill table
        self.table_model.update_data(self.type_counts)

        # Fill combo
        self.type_combo.clear()
        types = self.table_model._data["Triplet Type"].tolist()
        self.type_combo.addItems(types)
        self.type_combo.setMaxVisibleItems(10)
        self.type_combo.setEnabled(True)

        # Show first type
        if types:
            self.on_type_selected(types[0])

    def on_type_selected(self, triplet_type):
        if not triplet_type or not self.valid_triplets:
            return
        # Get the precomputed set of residues for this type
        if triplet_type in self.html_cache:
            html = self.html_cache[triplet_type]
        else:
            residue_set = self.type_to_residues.get(triplet_type, set())
            if not residue_set:
                return

            atom_spheres, label_positions = get_residue_sphere_coords_from_set(
                self.atom_info, self.coords, residue_set
            )
            html = render_py3dmol(self.pdb_string, atom_spheres, label_positions)
            self.html_cache[triplet_type] = html

        self.web_view.setHtml(html)


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('fusion')
    app.setStyleSheet('QComboBox {combobox-popup: 0}')  # this shows the normal scrollable list-view with max visible items
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
