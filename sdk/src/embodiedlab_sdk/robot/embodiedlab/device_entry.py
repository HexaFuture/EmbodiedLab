"""EmbodiedLab's optional native simulator adapter entry point.

The standalone EmbodiedLab entry point does not import or require this module.
The adapter supplies the same collision/held-body geometry available to the
historical EmbodiedLab simulator controller, without exposing it as Agent tools.
"""
from .native_geometry import NativeGeometry


def main():
    from server import main as benchmark_main
    benchmark_main(host_observer_factory=NativeGeometry)


if __name__ == '__main__':
    main()
