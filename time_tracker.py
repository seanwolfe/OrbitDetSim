import numpy as np
import math




class SimTime:
    def __init__(self, configs, current_od_index=None, current_epoch=None, iod_time=None,
                 current_integration_epoch=None, current_integration_index=None, attitude_coordination_expected_time=None):
        self.curr_od_index = current_od_index
        self.curr_epoch = current_epoch  # jdtdb - actual time in simulation
        self.curr_integration_epoch = current_integration_epoch  # jdtdb - time from which integrations begin (to get to curr epoch)
        self.curr_integration_index = current_integration_index  # index in actually minimoon trajectory file from which we are starting
        self.end_time = self.curr_epoch + configs['od_duration_days']  # jdtdb
        # epochs for attitude coordination
        epochs = configs.get("epochs", {})
        self.attcoord_dtmin = float(epochs.get("dt_min", 10.0))  # seconds
        self.attcoord_dtmax = float(epochs.get("dt_max", 600.0))  # seconds
        self.attcoord_numdt = int(epochs.get("n_dt", 60))
        self.attcoord_searchtimes = np.linspace(self.attcoord_dtmin, self.attcoord_dtmax,
                                                self.attcoord_numdt)  # seconds
        self.attcoord_searchtimes_jdtdb = None
        self.attcoord_time = None  # seconds
        self.attcoord_expectedtime = len(self.attcoord_searchtimes) * float(configs['epochs']['average_epoch_time']) \
            if attitude_coordination_expected_time is None else attitude_coordination_expected_time
        self.datacollect_time = int(configs['number_of_frames']) * float(configs['time_between_frames'])  # seconds
        self.detection_patchtime = configs['patch_time']
        big_H, big_L = int(configs['number_of_pixels'][0]), int(configs['number_of_pixels'][1])
        h, l = int(configs['patch_size'][0]), int(configs['patch_size'][1])
        self.detection_time = (int(math.ceil((big_H * big_L) / (h * l))) + 1) * self.detection_patchtime
        self.iod_time = iod_time
        self.slew_time = None  # seconds
        self.od_time = None


    def set_attcoord_searchtimes(self, od_time=None):
        if od_time is None:
            delta = (self.datacollect_time + self.detection_time + self.iod_time + self.attcoord_expectedtime) / 86400.0  # in days
        else:
            self.od_time = od_time
            delta = (
                            self.datacollect_time + self.detection_time + od_time + self.attcoord_expectedtime) / 86400.0  # in days
        big_t_set_jdtdb = self.curr_epoch + (self.attcoord_searchtimes / 86400.0) + delta
        big_t_set_jdtdb = big_t_set_jdtdb[big_t_set_jdtdb <= self.end_time + 1e-15]
        self.attcoord_searchtimes_jdtdb = big_t_set_jdtdb
        num_t_steps = len(big_t_set_jdtdb)
        self.attcoord_searchtimes = self.attcoord_searchtimes[:num_t_steps]

        return

    def set_attcoord_time(self, time):
        self.attcoord_time = time
        return

    def set_slew_time(self, time):
        self.slew_time = time

    def step(self, epoch, best_anchor_index, best_anchor_epoch):
        # update curr epoch
        self.curr_epoch = epoch
        self.curr_od_index += 1
        self.curr_integration_epoch = best_anchor_epoch
        self.curr_integration_index = best_anchor_index

    def check_search_times(self):
        stop = False
        od_time = 0 if self.od_time is None else self.od_time
        delta = (
                        self.datacollect_time + self.detection_time + od_time + self.attcoord_expectedtime) / 86400.0  # in days
        big_t_set_jdtdb = self.curr_epoch + (self.attcoord_searchtimes / 86400.0) + delta
        big_t_set_jdtdb = big_t_set_jdtdb[big_t_set_jdtdb <= self.end_time + 1e-15]
        num_t_steps = len(big_t_set_jdtdb)
        if num_t_steps == 0:
            stop = True
        return stop





