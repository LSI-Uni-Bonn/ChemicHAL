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

    generic_core_scaffold = MurckoScaffold.MakeScaffoldGeneric(original_core)
    # make representation more generic
    generic_core_scaffold = MurckoScaffold.GetScaffoldForMol(generic_core_scaffold)
    generic_core_scaffold = get_reduced_skeleton(generic_core_scaffold)

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

        sub_sites = [list(core_match).index(i) for i in find_all_tagged_atoms(generic_core_scaffold, n_subs)]
        sub_atoms_core = [example_core_scaffold.GetAtomWithIdx(i).GetIntProp("orig_idx") for i in sub_sites]

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
