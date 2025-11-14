"""
Factory for creating tight-binding models.
"""
from models.base_model import TightBindingModel
from models.parabolic_model import ParabolicModel
from models.square_lattice import SquareLatticeModel
from models.graphene import GrapheneModel
from models.advanced_lattices import CubicLatticeModel, AnisotropicLatticeModel, HoneycombLatticeModel
from models.true_3d_models import True3DCubicModel, True3DAnisotropicModel, True3DComplexModel
from models.multiband_3d_models import MultiBand3DTightBinding, ThreeBand3DKagome, QuadrupleBand3DModel
from models.ute2_model import UTe2Model, UTe2SimplifiedModel


def create_model(model_type: str, **kwargs) -> TightBindingModel:
    """
    Create a tight-binding model instance.
    
    Args:
        model_type: Type of model ("parabolic", "square_lattice", "graphene", "cubic_3d", "anisotropic", "honeycomb", "true_3d_cubic", "true_3d_aniso", "true_3d_complex")
        **kwargs: Model-specific parameters
        
    Returns:
        TightBindingModel instance
    """
    if model_type == "parabolic":
        return ParabolicModel(**kwargs)
    elif model_type == "square_lattice":
        return SquareLatticeModel(**kwargs)
    elif model_type == "graphene":
        return GrapheneModel(**kwargs)
    elif model_type == "cubic_3d":
        return CubicLatticeModel(**kwargs)
    elif model_type == "anisotropic":
        return AnisotropicLatticeModel(**kwargs)
    elif model_type == "honeycomb":
        return HoneycombLatticeModel(**kwargs)
    elif model_type == "true_3d_cubic":
        return True3DCubicModel(**kwargs)
    elif model_type == "true_3d_aniso":
        return True3DAnisotropicModel(**kwargs)
    elif model_type == "true_3d_complex":
        return True3DComplexModel(**kwargs)
    elif model_type == "multiband_2band":
        return MultiBand3DTightBinding(**kwargs)
    elif model_type == "multiband_3band":
        return ThreeBand3DKagome(**kwargs)
    elif model_type == "multiband_4band":
        return QuadrupleBand3DModel(**kwargs)
    elif model_type == "ute2_full":
        return UTe2Model(**kwargs)
    elif model_type == "ute2_simplified":
        return UTe2SimplifiedModel(**kwargs)
    # Add more models here as needed
    else:
        available = "parabolic, square_lattice, graphene, cubic_3d, anisotropic, honeycomb, true_3d_cubic, true_3d_aniso, true_3d_complex, multiband_2band, multiband_3band, multiband_4band, ute2_full, ute2_simplified"
        raise ValueError(f"Unknown model type: {model_type}. Available: {available}")


def list_available_models():
    """List all available tight-binding models."""
    models = {
        "parabolic": "Parabolic dispersion (backward compatible)",
        "square_lattice": "Square lattice with nearest-neighbor hopping",
        "graphene": "Graphene with two sublattices",
        "cubic_3d": "3D cubic lattice (2D slice visualization)",
        "anisotropic": "Anisotropic lattice with different tx, ty",
        "honeycomb": "Honeycomb lattice with two bands",
        "true_3d_cubic": "True 3D cubic lattice (full 3D Fermi surface)",
        "true_3d_aniso": "True 3D anisotropic lattice (full 3D Fermi surface)", 
        "true_3d_complex": "True 3D complex lattice (multiple hopping terms)",
        "multiband_2band": "Two-band 3D model with hybridization (electron/hole pockets)",
        "multiband_3band": "Three-band 3D Kagome-like model (flat bands + Dirac points)",
        "multiband_4band": "Four-band 3D d-p model (transition metal compounds)",
        "ute2_full": "UTe₂ full 5-orbital model (U 5f + Te 5p physics)",
        "ute2_simplified": "UTe₂ simplified 3-band model (U-dominant physics)"
    }
    
    print("Available tight-binding models:")
    for name, description in models.items():
        print(f"  {name}: {description}")