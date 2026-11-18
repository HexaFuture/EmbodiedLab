"""Reaction bench physics device: the chemistry backend with the reaction apparatus (the kinetics bench plus the dry
block). Every physics tick the bench also learns which cuvette stands in each block well (the kit's seat rule); the
block's controller screen (SET, BLOCK, the LOG mark) is redrawn when what it shows changes."""
import thermal_lcd as L
from chem_backend import ChemBackend


class ReactionBackend(ChemBackend):
    ASSET = 'reaction_apparatus'

    def _build(self, asset):
        """ChemBackend._build, keeping the block's screen panel."""
        import chem_render
        import reaction_model as RM
        from chem_solver import Dye
        RM.register_spectra()
        bench = self.configuration['bench']
        dyes = {name: Dye(name, factor) for name, factor in bench['dyes'].items()}
        built = asset.build(self.stage, {
            'bottle_mL': {n: s['volume_mL'] for n, s in bench['sources'].items()},
            'bottle_rgb': {n: chem_render.linear_rgb(s['composition_M'], dyes, path_cm=asset.bottle_inner(s['volume_mL'])[0] * 100).tolist()
                           for n, s in bench['sources'].items()},
            'bottle_fill': {n: 1.0 for n in bench['sources']},
            'labels': bench.get('labels'), 'extra_key': bench.get('extra_key')})
        self.panels = built['panels']
        self.panels_drawn = {name: None for name in self.panels}
        return {'objects': built['cuvettes'], 'buttons': built['buttons'], 'seats': built['seats']}

    def _support_paths(self, key, built):
        return super()._support_paths(key, built) + self.asset.block_support_paths()

    def _world(self, positions, orientations):
        world, depth, tilts = super()._world(positions, orientations)
        world['block'] = {w: world['seats'].get(f'block:{w}') for w in range(1, 5)}
        return world, depth, tilts

    def _redraw(self):
        super()._redraw()
        if self.bench is None:
            return
        shown = self.bench.block_display()
        state = self.asset.controller_state(shown['set_C'], shown['block_C'], shown.get('log', False))
        if state != self.panels_drawn['block']:
            path, rows = self.panels['block']
            self.panels_drawn['block'] = L.show(self.stage, path, rows, state, self.panels_drawn['block'])
            self.history_stale = True
