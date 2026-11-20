"""Public apparatus descriptions, shared by every participant system.

These are fixed construction/label facts, not localization, a motion plan,
live object poses, private parameters, or an experimental solution.
"""


def apparatus_description(calibration):
    return {
        'identification': {
            'moving_bob': 'Small gold/brass sphere at the lower free end of each thin pendulum rod.',
            'fixed_parts': 'The dark upper clamp, white angle plate, bearing housing, support pole and optical-gate frame are fixed apparatus, not moving bobs.',
        },
        'motion': {
            'constraint': 'Each pendulum swings in one vertical plane about its bearing.',
        },
        'instrument_labels': [
            {'label': f'P{index}', 'pendulum_id': key,
             'effective_length_m': length,
             'matching_timer': 'The timer with the same P label measures this pendulum.'}
            for index, (key, length) in enumerate(sorted(calibration.items(), key=lambda item: item[1]), 1)
        ],
        'observation': 'Labels are apparatus identities, not target pixels or robot poses.',
    }
