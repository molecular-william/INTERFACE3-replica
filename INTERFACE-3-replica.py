import streamlit as st
import pandas as pd
import numpy as np
from Bio import PDB
import alphashape
from collections import defaultdict, Counter
import tempfile
import io

# -------------------------------------------------------------------
# Helper functions (adapted from the original code)
# -------------------------------------------------------------------

def _parse_helix_record(line: str) -> dict:
    """Parse a HELIX record from a PDB file."""
    data = line.split()
    helix_id = int(data[2])
    chain = data[4]
    start = int(data[5])
    end = int(data[8])
    return {'helix_id': helix_id, 'chain_id': chain,
            'start_residue': start, 'end_residue': end}


def parse_pdb_with_helix(pdb_path, atom_type='heavy'):
    """
    Parse a PDB file, extract atomic coordinates and metadata,
    and assign helix IDs based on HELIX records.
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)

    # Build helix map from HELIX records in the file
    helix_map = {}
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('HELIX'):
                rec = _parse_helix_record(line)
                for resnum in range(rec['start_residue'], rec['end_residue'] + 1):
                    helix_map[(rec['chain_id'], resnum)] = rec['helix_id']

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
    """Compute all atomic contact edges from the alpha shape."""
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
    """Build adjacency list from edge set."""
    adj = [set() for _ in range(n_atoms)]
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
    return adj


def find_triangles(adj):
    """Find all triangles (3 mutually connected atoms) in the graph."""
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
    """Convert three‑letter residue code to one‑letter code."""
    three_to_one = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
        'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
    }
    return three_to_one.get(resname.upper(), 'X')


def filter_triangles(triangles, atom_info):
    """
    Keep only triangles where:
      - three atoms belong to three different residues,
      - all residues are in helices (helix_id not None),
      - residues come from at least two different helices.
    Returns list of dicts with 'atoms' and 'triplet_type'.
    """
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


# -------------------------------------------------------------------
# Cached computation (heavy part)
# -------------------------------------------------------------------
@st.cache_data(show_spinner="Computing triplets...")
def compute_triplets(pdb_bytes, atom_type, alpha):
    """
    Run the full pipeline on the uploaded PDB file.
    Returns:
        pdb_string : str (content of PDB file for visualisation)
        atom_info  : list of dict
        coords     : np.ndarray
        valid_triplets : list of dict
        type_counts : dict {triplet_type: count}
    """
    # Write bytes to a temporary file for Bio.PDB and HELIX parsing
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.pdb', delete=False) as tmp:
        tmp.write(pdb_bytes)
        tmp_path = tmp.name

    # Read PDB content as string for later visualisation
    pdb_string = pdb_bytes.decode('utf-8')

    # Parse structure and extract atom info
    coords, atom_info = parse_pdb_with_helix(tmp_path, atom_type)

    if len(coords) < 4:
        st.error("Need at least 4 atoms. Cannot compute alpha shape.")
        return pdb_string, atom_info, coords, [], {}

    # Compute edges
    edges = compute_alpha_edges(coords, alpha)

    # Find triangles
    adj = build_adjacency(edges, len(coords))
    triangles = find_triangles(adj)

    # Filter triplets
    valid_triplets = filter_triangles(triangles, atom_info)

    # Count types
    type_counts = Counter(t['triplet_type'] for t in valid_triplets)

    return pdb_string, atom_info, coords, valid_triplets, type_counts


# -------------------------------------------------------------------
# Visualisation with py3Dmol
# -------------------------------------------------------------------
def get_residue_sphere_coords(atom_info, coords, triplet_type, valid_triplets):
    """
    For a given triplet type, find all residues involved.
    Returns:
        atom_spheres : list of (chain, resnum, resname, atom_name, element, x, y, z)
        label_positions : list of (chain, resnum, resname, x, y, z) for CA (or first atom)
    """
    # Identify residues to show
    residues_to_show = set()
    for trip in valid_triplets:
        if trip['triplet_type'] == triplet_type:
            i, j, k = trip['atoms']
            residues_to_show.add((atom_info[i]['chain'], atom_info[i]['resnum']))
            residues_to_show.add((atom_info[j]['chain'], atom_info[j]['resnum']))
            residues_to_show.add((atom_info[k]['chain'], atom_info[k]['resnum']))

    atom_spheres = []
    label_candidates = {}  # (chain, resnum) -> (resname, coord, atom_name)

    for idx, info in enumerate(atom_info):
        key = (info['chain'], info['resnum'])
        if key in residues_to_show:
            # Add atom sphere with element
            atom_spheres.append((
                info['chain'],
                info['resnum'],
                info['resname'],
                info['atom_name'],
                info['element'],           # now includes element
                coords[idx][0],
                coords[idx][1],
                coords[idx][2]
            ))

            # Update label candidate (prefer CA)
            if key not in label_candidates:
                label_candidates[key] = (info['resname'], coords[idx], info['atom_name'])
            else:
                if info['atom_name'] == 'CA' and label_candidates[key][2] != 'CA':
                    label_candidates[key] = (info['resname'], coords[idx], info['atom_name'])

    # Build label positions
    label_positions = []
    for (chain, resnum), (resname, coord, _) in label_candidates.items():
        label_positions.append((chain, resnum, resname, coord[0], coord[1], coord[2]))

    return atom_spheres, label_positions


def render_py3dmol(pdb_string, atom_spheres, label_positions):
    """
    Generate an HTML string with py3Dmol view.
    Spheres are coloured and sized by element.
    """
    import py3Dmol

    # Element‑based colour scheme (CPK‑like)
    color_map = {
    'C': 'gray', 'N': 'blue','O': 'red','S': 'yellow',
    'H': 'white', 'CA': 'green','CB': 'gray','CD': 'gray',
    'CG': 'gray','CE': 'gray','NZ': 'blue','OD1': 'red',
    'OD2': 'red', 'NE': 'blue', 'NH1': 'blue', 'NH2': 'blue',
    'OE1': 'red', 'OE2': 'red', 'ND2': 'blue', 'OG': 'red',
    'OG1': 'red', 'SG': 'yellow', 'NE2': 'blue', 'ND1': 'blue',
    'CE1': 'gray', 'CD2': 'gray', 'CG1': 'gray', 'CG2': 'gray',
    'CD1': 'gray', 'CE2': 'gray', 'CZ': 'gray', 'OH': 'red',
    'NE1': 'blue', 'CZ2': 'gray', 'CZ3': 'gray', 'CE3': 'gray',
    'CH2': 'gray', 'SD': 'yellow', 'DEFAULT': 'gray',
    }

    # Radius per element (van der Waals radii scaled for visibility)
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

    # Add spheres for all atoms
    for (chain, resnum, resname, atom_name, element, x, y, z) in atom_spheres:
        # Use element for colour and radius, fallback to defaults
        colour = color_map.get(element, color_map['DEFAULT'])
        radius = radius_map.get(element, radius_map['DEFAULT'])

        view.addSphere({
            'center': {'x': float(x), 'y': float(y), 'z': float(z)},
            'radius': radius,
            'color': colour,
            'alpha': 0.98
        })

    # Add labels at CA positions
    for (chain, resnum, resname, x, y, z) in label_positions:
        view.addLabel(f"{chain}:{resnum} {resname}", {
            'position': {'x': float(x), 'y': float(y), 'z': float(z)},
            'fontSize': 12,
        })

    view.zoomTo()
    return view._make_html()


# -------------------------------------------------------------------
# Streamlit App
# -------------------------------------------------------------------
st.set_page_config(page_title="Membrane Protein Triplet Visualiser", layout="wide")
st.title("🔬 Membrane Protein Triplet Extractor & Visualiser")
st.markdown("""
Upload a PDB file of a membrane protein.  
The app will compute all residue triplets (three different residues from at least two different helices)  
that are in contact according to an **alpha‑shape** of the atomic coordinates.
""")

# Sidebar
with st.sidebar:
    st.header("Input Parameters")
    uploaded_file = st.file_uploader("Choose a PDB file", type=['pdb'])
    atom_type = st.radio("Atom selection", options=['heavy', 'all'],
                         help="'heavy' excludes hydrogens; 'all' includes all atoms.")
    alpha = st.number_input("Alpha value (Å)", min_value=1.0, max_value=8.0, value=3.0, step=0.2)
    compute_btn = st.button("Compute triplets", type="primary")

# Main area (after sidebar)
if uploaded_file is not None and compute_btn:
    pdb_bytes = uploaded_file.getvalue()

    # Run computation (cached)
    pdb_string, atom_info, coords, valid_triplets, type_counts = compute_triplets(
        pdb_bytes, atom_type, alpha
    )

    # Store in session state
    st.session_state['pdb_string'] = pdb_string
    st.session_state['atom_info'] = atom_info
    st.session_state['coords'] = coords
    st.session_state['valid_triplets'] = valid_triplets
    st.session_state['type_counts'] = type_counts

# Always display results if they exist in session state
if 'valid_triplets' in st.session_state and st.session_state['valid_triplets']:
    st.subheader("Results")
    st.write(f"**Number of atoms selected:** {len(st.session_state['coords'])}")
    st.write(f"**Valid triplets found:** {len(st.session_state['valid_triplets'])}")

    # Show a table of triplet types and counts
    st.write("**Triplet type counts:**")
    type_counts = st.session_state['type_counts']
    type_df = pd.DataFrame(list(type_counts.items()), columns=['Triplet Type', 'Count']).sort_values(by='Count', ascending=False)
    st.dataframe(type_df, use_container_width=True)

    # Dropdown for selecting a triplet type
    triplet_types = sorted(type_counts.keys())
    selected_type = st.selectbox("Choose a triplet type to visualise", triplet_types)

    if selected_type:
        atom_spheres, label_positions = get_residue_sphere_coords(
            st.session_state['atom_info'],
            st.session_state['coords'],
            selected_type,
            st.session_state['valid_triplets']
        )
        st.write(f"Highlighting **{len(atom_spheres)} atoms** from **{len(label_positions)} residues** involved in type **{selected_type}**.")

        html = render_py3dmol(st.session_state['pdb_string'], atom_spheres, label_positions)
        st.components.v1.html(html, height=650)

elif uploaded_file is not None and not compute_btn:
    st.info("Click 'Compute triplets' in the sidebar to start the analysis.")
else:
    st.info("Please upload a PDB file.")