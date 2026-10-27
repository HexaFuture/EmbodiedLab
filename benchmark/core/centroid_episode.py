"""torsion_centroid episode on the torsion bench (recorder panel, damper lever, storage box, locating pin)."""
from copy import deepcopy
from instance_episodes import InstanceMixin, TORSION_TOOLS, torsion_mark_colour
from centroid_model import mass_properties
import math


def centroid_dynamics(instance):
    h = instance['hidden']
    props = mass_properties(h['ballast_fraction'], h['imbalance'], h['ballast_halfspan_m'])
    if abs(props['com_m'][0]-h['x_com_m']) > 1e-12 or abs(props['diagonal_inertia_kgm2'][2]-h['I_com_kgm2']) > 1e-12:
        raise ValueError('specimen mass model and instance disagree')
    k, i0 = h['kappa_Nm_per_rad'], h['I0_kgm2']
    return dict(kappa_nm_per_rad=k, turntable_inertia_kgm2=i0, ballast_fraction=h['ballast_fraction'],
                imbalance=h['imbalance'], ballast_halfspan_m=h['ballast_halfspan_m'], damping_nms_per_rad=1.2e-4,
                brake_damping_nms_per_rad=2*math.sqrt(k*(i0+.004)),
                mark_rgb=torsion_mark_colour(instance['instance_id']))


def centroid_episode(backend, instance, output, *, physics_hz, horizon_ticks):
    from device_runtime import DeviceRuntime
    from centroid_gate import CentroidPhotogate
    import centroid_apparatus as asset

    class CentroidEpisode(InstanceMixin, DeviceRuntime):
        def public_task(self):
            return self.instance_public_task({})

        def robot_state(self):
            return super().robot_state() | {'gripper_holding': bool(self.backend.robot_state()['gripper_holding'])}

        def seal_payload(self):
            """Terminal state, device, evidence and files of the sealed record (also used at stop)."""
            gate = self.backend.photogate
            evidence = {eid: dict(public=deepcopy(record), kind='centroid_photogate', **deepcopy(gate.private[eid]))
                        for eid, record in gate.records.items()}
            return (
                {'loads': self.backend.loads(), 'mount': list(self.backend.mount()),
                 'photogate_active': gate.active is not None, **self.backend.bench_state('seal')},
                {'standard_disc_inertia_kgm2': asset.STANDARD_INERTIA_KGM2,
                 'object_masses_kg': {k: v[2] for k, v in asset.OBJECTS.items()},
                 'holes_m': {str(n): x for n, x in asset.HOLES.items()},
                 'flag_half_angle_rad': asset.FLAG_HALF_ANGLE_RAD,
                 'storage_box': {'seats': {str(i + 1): key for i, key in enumerate(asset.BOX_SEATS)},
                                 'dxy_m': asset.SEAT_DXY_M, 'abs_dz_m': asset.SEAT_DZ_M,
                                 'tilt_deg': asset.SEAT_TILT_DEG}}, evidence)

        def process_extra(self):
            """Storage box occupancy for the process record: at start, seat_in / seat_out events, at stop."""
            return {'storage_box_seats': self.backend.end_seat_log()}

        def _dispatch(self, name, args):
            if self.terminated:
                raise ValueError('episode terminated')
            if not isinstance(args, dict):
                raise ValueError('arguments must be an object')
            gate = self.backend.photogate
            if name == 'get_task_spec' and args == {}:
                return self.public_task()
            if name == 'export_photogate' and set(args) == {'evidence_id'}:
                return gate.export(args['evidence_id'])
            if name in TORSION_TOOLS:
                raise ValueError('unexpected arguments')
            if name == 'submit_result':
                return self.freeze_and_seal(args, *self.seal_payload())
            return super()._dispatch(name, args)

        def evaluation(self):
            return self.instance_evaluation()

    # Apparatus preparation (not charged): open gripper, settle with the preparation damping, then release it.
    backend.set_gripper(.08)
    for _ in range(130 * (physics_hz // 30)):
        backend.step(None)
        backend.after_step({}, backend.tick_index)
    if not backend.at_rest() or backend.loads():
        raise RuntimeError('empty turntable failed to settle')
    backend.tick_index = 0
    backend.history.clear()
    episode = CentroidEpisode(backend, physics_hz=physics_hz, horizon_ticks=horizon_ticks)
    episode._instance_init(instance, output)
    backend.release_settling_damping()
    backend.photogate = CentroidPhotogate(episode.episode_id, physics_hz, backend.mount, asset.RECORD_S_RANGE)
    backend.start_seat_log()
    backend.bench_state('start')
    return episode
