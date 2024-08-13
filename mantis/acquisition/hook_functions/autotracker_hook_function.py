from mantis.acquisition.autotracker import Autotracker


def autotracker_hook_fn(axes, dataset) -> None:
    """
    Pycromanager hook function that is called when an image is saved.

    Parameters
    ----------
    axes : Position, Time, Channel, Z_slice
    dataset: Dataset saved in disk
    """

    # Get reference to the acquisition engine and it's settings
    # TODO: This is a placeholder, the actual implementation will be different
    acq = "reference to the acquisition engine"
    shift_limit = acq.autofocus_settings.shift_limit
    tracking_method = acq.autofocus_settings.tracking_method
    output_shift_path = './output.csv'

    # Get axes info
    p_idx = axes['position']
    t_idx = axes['time']
    channel = axes['channel']
    z_idx = axes['z']

    # Logic to get the volumes
    # TODO: This is a placeholder, the actual implementation will be different
    volume_t0_axes = (p_idx, t_idx, channel, z_idx)
    volume_t1_axes = (p_idx, t_idx, channel, z_idx)

    volume_t0 = dataset.read_image(**volume_t0_axes)
    volume_t1 = dataset.read_image(**volume_t1_axes)

    # Compute the shifts
    tracker = Autotracker(
        autofocus_method=tracking_method,
        shift_limit=shift_limit,
        output_shifts_path=output_shift_path,
    )
    # Reference and moving volumes
    tracker.estimate_shift(volume_t0, volume_t1)

    # Save the shifts
    # TODO: This is a placeholder, the actual implementation will be different

    # Update the event coordinates
    # TODO: This is a placeholder, the actual implementation will be different
