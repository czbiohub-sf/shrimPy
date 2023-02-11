from pycromanager import Core
import time

def autofocus_fn(ZMQ_PORT, z_stage, z_position_list, events):
    """ Intended to be used as post-hardware hook function, wrapped by functools.partial

    """

    if not hasattr(autofocus_fn, 'pos_index'):
        autofocus_fn.pos_index = -1

    if isinstance(events, list):
        event = events[0]  # acquisition is sequenced
    else:
        event = events  # acquisition is not sequenced


    if 'axes' in event.keys():  # some events only modify hardware
        # Only run autofocus at new positions (including the first one)
        pos_index = event['axes']['position']
        if pos_index != autofocus_fn.pos_index:
            # get the Micro-Manager core
            mmc = Core(port=ZMQ_PORT)

            # apply ZDrive position
            mmc.set_position(z_stage, z_position_list[pos_index])
            time.sleep(2)  # wait for oil to catch up

            # apply autofocus
            z_old = mmc.get_position(z_stage)
            print(f'Position before autofocus: {z_old}')

            z_offsets = [-5, 5, -10, 10]
            for z_offset in z_offsets:
                try:
                    print('Calling autofocus.')
                    mmc.full_focus()
                except:
                    print(f'Autofocus failed. Applying offset {z_offset}.')
                    mmc.set_relative_position(z_stage, z_offset)
                else:
                    print('Autofocus successfully engaged.')
                    break

            # # autofocus
            # print('Calling autofocus')
            # try:
            #     mmc.full_focus()
            # except:
            #     print('First try failed. Move +5 um and retry')
            #     mmc.set_relative_position(z_stage, 5)  # try moving up
            #     try:
            #         mmc.full_focus()
            #     except:
            #         print('Second try failed. Move -10 um and retry')
            #         mmc.set_relative_position(z_stage, -10)  # try moving down
            #         try:
            #             mmc.full_focus()
            #         except:
            #             print('Autofocus failed')
            #         else:
            #             print('Autofocus engaged!!')
            #     else:
            #         print('Autofocus engaged!!')
            # else:
            #     print('Autofocus engaged!!')

            z_new = mmc.get_position(z_stage)
            print(f'Position after autofocus: {z_new}')

        # Keep track of last time point on which autofocus ran
        autofocus_fn.pos_index = pos_index

    return event
