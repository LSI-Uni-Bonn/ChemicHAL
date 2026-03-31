from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

def get_reduced_skeleton(mol: Chem.Mol) -> Chem.Mol:
    """
    Return the reduced scaffold of a cyclic skeleton
    """
    edit_mol = Chem.RWMol(mol)
    to_remove = [True]
    while to_remove:
        to_remove = []
        for a in edit_mol.GetAtoms():
            if a.GetDegree() == 2:
                if a.IsInRing() is False:
                    n1,n2 = [n.GetIdx() for n in a.GetNeighbors()]
                    if edit_mol.GetBondBetweenAtoms(n1,n2) is None:
                        to_remove.append(a.GetIdx())
                        edit_mol.AddBond(n1,n2,Chem.BondType.SINGLE)
        for a in reversed(to_remove):
            edit_mol.RemoveAtom(a)
        edit_mol.UpdatePropertyCache()
        Chem.SanitizeMol(edit_mol)

    return Chem.Mol(edit_mol)

def sanitize_clean_protonate_mol(mol):
    du = Chem.MolFromSmiles('*')
    mol = AllChem.ReplaceSubstructs(mol, du, Chem.MolFromSmiles('[H]'), True)[0]
    mol = Chem.RemoveHs(mol)
    mol.UpdatePropertyCache(strict=True)
    Chem.SanitizeMol(mol)

    return mol

def tag_substitution_atom(mol, tag = "sub_tag"):
    mol = Chem.RWMol(mol)
    attach_points = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            neighbour = atom.GetNeighbors()[0]
            attach_points.append(neighbour.GetIdx())
    for i, idx in enumerate(attach_points):
        mol.GetAtomWithIdx(idx).SetProp(tag, str(i))
    return mol

def find_all_tagged_atoms(mol, n_subs, tag = "sub_tag"):
    found = []
    for i in range(n_subs):
        for atom in mol.GetAtoms():
            if atom.HasProp(tag):
                if atom.GetProp(tag) == str(i):
                    found.append(atom.GetIdx())
    return found


def annotate_sub_sites(core, scaffolds, n_subs):

    assert len(scaffolds) != 0

    original_core = tag_substitution_atom(core)

    # Build the generic scaffold before reduction so sub_tags are still present.
    generic_core_unreduced = MurckoScaffold.MakeScaffoldGeneric(original_core)
    generic_core_unreduced = MurckoScaffold.GetScaffoldForMol(generic_core_unreduced)

    # Label every atom with its pre-reduction index so we can trace survivors.
    rw_pre = Chem.RWMol(generic_core_unreduced)
    for atom in rw_pre.GetAtoms():
        atom.SetIntProp("_pre_idx", atom.GetIdx())
    generic_core_unreduced = Chem.Mol(rw_pre)

    # Reduced skeleton used for substructure matching.
    generic_core_scaffold = get_reduced_skeleton(Chem.RWMol(generic_core_unreduced))

    # Build mapping: sub_tag value (0..n_subs-1) -> atom idx in generic_core_scaffold.
    # For tags that survived reduction the atom is found directly.
    # For tags removed by reduction we walk the pre-reduction neighbors and find
    # which neighbor survived into the reduced skeleton (by _pre_idx).
    pre_idx_to_reduced = {
        atom.GetIntProp("_pre_idx"): atom.GetIdx()
        for atom in generic_core_scaffold.GetAtoms()
        if atom.HasProp("_pre_idx")
    }

    sub_tag_to_reduced = {}
    linker_tags = set()  # tag values whose original atom was removed by reduction
    for tag_val in range(n_subs):
        # Check if the tag survived directly.
        for atom in generic_core_scaffold.GetAtoms():
            if atom.HasProp("sub_tag") and atom.GetProp("sub_tag") == str(tag_val):
                sub_tag_to_reduced[tag_val] = atom.GetIdx()
                break
        if tag_val in sub_tag_to_reduced:
            continue
        # Tag was removed by reduction — find the pre-reduction atom and trace
        # to a surviving neighbor whose topology is unchanged.
        for atom in generic_core_unreduced.GetAtoms():
            if atom.HasProp("sub_tag") and atom.GetProp("sub_tag") == str(tag_val):
                for nb in atom.GetNeighbors():
                    nb_pre = nb.GetIntProp("_pre_idx")
                    if nb_pre in pre_idx_to_reduced:
                        sub_tag_to_reduced[tag_val] = pre_idx_to_reduced[nb_pre]
                        linker_tags.add(tag_val)
                        break
                break

    if len(sub_tag_to_reduced) != n_subs:
        return []  # cannot locate all substitution sites on the reduced skeleton

    subbed_scaffolds = []
    skipped_cores = []

    for f, example_core in enumerate(scaffolds):

        example_core = sanitize_clean_protonate_mol(example_core)

        for atom in example_core.GetAtoms():
            atom.SetIntProp("orig_idx", atom.GetIdx())

        example_core_scaffold = MurckoScaffold.MakeScaffoldGeneric(example_core)
        # make representation more generic
        example_core_scaffold = MurckoScaffold.GetScaffoldForMol(example_core_scaffold)
        example_core_scaffold = get_reduced_skeleton(example_core_scaffold)

        core_match = generic_core_scaffold.GetSubstructMatch(example_core_scaffold, useChirality=True)

        if len(core_match) == 0:
            continue

        try:
            sub_sites = [list(core_match).index(sub_tag_to_reduced[tag_val])
                         for tag_val in range(n_subs)]
        except ValueError:
            continue  # a substitution-site atom is not part of this match

        # For each tag, get the orig_idx in the full example_core molecule.
        # For linker_tags the mapped atom is a ring atom (the surviving neighbor of
        # the removed linker); the correct attachment point is the degree-2 non-ring
        # neighbor of that ring atom in the full molecule — i.e. the linker atom that
        # exists in the contrastive scaffold but was also collapsed by reduction.
        sub_atoms_core = []
        for tag_val, site_idx in zip(range(n_subs), sub_sites):
            orig_idx = example_core_scaffold.GetAtomWithIdx(site_idx).GetIntProp("orig_idx")
            if tag_val in linker_tags:
                # Walk neighbors in the full example_core for a degree-2 non-ring atom.
                linker_idx = None
                for nb in example_core.GetAtomWithIdx(orig_idx).GetNeighbors():
                    if nb.GetDegree() == 2 and not nb.IsInRing():
                        linker_idx = nb.GetIdx()
                        break
                sub_atoms_core.append(linker_idx if linker_idx is not None else orig_idx)
            else:
                sub_atoms_core.append(orig_idx)

        rwm = Chem.RWMol(example_core)

        skip_core = False

        for idx in sub_atoms_core:

            atom = rwm.GetAtomWithIdx(idx)
            if atom.GetNumExplicitHs() + atom.GetNumImplicitHs() > 0:
                continue
            else:
                skip_core = True
                skipped_cores.append(example_core)
                break

        if skip_core:
            continue

        for i, idx in enumerate(sub_atoms_core):
            dummy = Chem.Atom(0)
            dummy.SetAtomMapNum(i+1)
            dummy_idx = rwm.AddAtom(dummy)
            rwm.AddBond(idx, dummy_idx,Chem.BondType.SINGLE)

            try:
                rwm.UpdatePropertyCache(strict=True)

            except:
                rwm.GetAtomWithIdx(idx).SetNumExplicitHs(0)
                rwm.GetAtomWithIdx(idx).SetNoImplicit(False)
                rwm.UpdatePropertyCache(strict=True)

        try:
            Chem.SanitizeMol(rwm)
            subbed_scaffolds.append(rwm)

        except:
            continue

    return subbed_scaffolds
