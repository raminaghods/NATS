'''
Code for submitted work:

``Multi Agent Active Search using Realistic Depth-Aware Noise Model'',
submitted to 2020 Conference on Robot Learning
author: anonymous

please do not distribute. The code will become public upon acceptance of the paper.

Manager for multiple agents(workers).

(structure is referenced from parallel Thompson Sampling by:
``@inproceedings{kandasamy2018parallelised,
  title={Parallelised bayesian optimisation via thompson sampling},
  author={Kandasamy, Kirthevasan and Krishnamurthy, Akshay and Schneider, Jeff and P{\'o}czos, Barnab{\'a}s},
  booktitle={International Conference on Artificial Intelligence and Statistics},
  pages={133--142},
  year={2018}
GitHub repository: {https://github.com/kirthevasank/gp-parallel-ts)}
}''

'''

import os
import shutil
import time
#from sets import Set
from multiprocessing import Process

import pickle as pkl

TIME_TOL = 1e-5

class WorkerManager(object):

    def __init__(self, func_caller, worker_ids, poll_time, trialnum):
        if hasattr(worker_ids, '__iter__'):
            self.worker_ids = worker_ids
        else:
            self.worker_ids = list(range(worker_ids))
        self.num_workers = len(self.worker_ids)
        self.poll_time = poll_time
        self.func_caller = func_caller
        # These will be set in reset
        self.optimiser = None
        self.latest_results = None
        # Reset
        self.reset(trialnum)

    def reset(self, trialnum):
        """ Resets everything. """
        self.optimiser = None
        self.latest_results = [] # A list of namespaces
        # Create the last receive times
        self.last_receive_times = {wid:0.0 for wid in self.worker_ids}

        self.result_dir_names = {wid:'./log/exp%d/result_%s'%(trialnum, str(wid)) for wid in
                                                      self.worker_ids}
        # Create the working directories
        self.working_dir_names = {wid:'./log/exp%d/working_%s/tmp'%(trialnum, str(wid)) for wid in
                                                            self.worker_ids}

        self._result_file_name = 'result.pkl'
        self._num_file_read_attempts = 100
        self._file_read_poll_time = 0.5 # wait for 0.5 seconds

        self._child_reset()


    def _child_reset(self):
        self._delete_and_create_dirs(list(self.result_dir_names.values()))
        self._delete_dirs(list(self.working_dir_names.values()))
        self.free_workers = set(self.worker_ids)
        self.qinfos_in_progress = {wid:None for wid in self.worker_ids}
        self.worker_processes = {wid:None for wid in self.worker_ids}

    def set_optimiser(self, optimiser):
        self.optimiser = optimiser


    def _delete_dirs(self, list_of_dir_names):
        """ Deletes a list of directories."""
        for dir_name in list_of_dir_names:
            if os.path.exists(dir_name):
                shutil.rmtree(dir_name)


    def _delete_and_create_dirs(self, list_of_dir_names):
        """ Deletes a list of directories and creates new ones. """
        for dir_name in list_of_dir_names:
            if os.path.exists(dir_name):
                shutil.rmtree(dir_name)
            os.makedirs(dir_name)

    def _get_result_file_name_for_worker(self, worker_id):
        """ Computes the result file name for the worker. """
        return os.path.join(self.result_dir_names[worker_id], self._result_file_name)

    def _read_result_from_file(self, result_file_name):
        """ Reads the result from the file name. """
        #pylint: disable=bare-except
        num_attempts = 0
        result = 0.5
        while num_attempts < self._num_file_read_attempts:
            try:
                # file_reader = open(result_file_name, 'r')
                # read_in = float(file_reader.read().strip())
                # file_reader.close()
                # result = read_in
                with open(result_file_name, 'rb') as f:
                    result = pkl.load(f)
                    # print("read result: ",result)
                break
            except:
                print('Encountered error %d times when reading %s. Trying again.'%(num_attempts,result_file_name))
                num_attempts += 1
                time.sleep(self._file_read_poll_time)
                #file_reader.close()
        return result


    def _read_result_from_worker_and_update(self, worker_id):
        """ Reads the result from the worker. """
#        print("reading result from worker: ",worker_id)
        # Read the file
        result_file_name = self._get_result_file_name_for_worker(worker_id)
        val = self._read_result_from_file(result_file_name)
        # Now update the relevant qinfo and put it to latest_results
        qinfo = self.qinfos_in_progress[worker_id]
        qinfo.val = val #dict of {x,y} from ThompsonActiveSearch
        # if not hasattr(qinfo, 'true_val'):
        #   qinfo.true_val = val
        qinfo.receive_time = self.optimiser.get_curr_spent_capital()
        qinfo.eval_time = qinfo.receive_time - qinfo.send_time
        self.latest_results.append(qinfo)

        # Update receive time
        self.last_receive_times[worker_id] = qinfo.receive_time
        # Delete the file.
        os.remove(result_file_name)
        # Delete content in a working directory.
        shutil.rmtree(self.working_dir_names[worker_id])
        # Add the worker to the list of free workers and clear qinfos in progress.
        self.worker_processes[worker_id].terminate()
        self.worker_processes[worker_id] = None
        self.qinfos_in_progress[worker_id] = None
        self.free_workers.add(worker_id)


    def fetch_latest_results(self):
        """ Returns the latest results. """
        ret_idxs = []
        for i in range(len(self.latest_results)):
            if (self.latest_results[i].receive_time <=
                self.optimiser.get_curr_spent_capital()):# + TIME_TOL):
                ret_idxs.append(i)
        keep_idxs = [i for i in range(len(self.latest_results)) if i not in ret_idxs]
        ret = [self.latest_results[i] for i in ret_idxs] #list of dicts {'x','y'} <- points returned by ThompsonActiveSearch for different agents
        self.latest_results = [self.latest_results[i] for i in keep_idxs]
        return ret


    def close_all_jobs(self):
        """ closes all jobs (TODO) """
        pass


    def _get_last_receive_time(self):
        """ Returns the last time we received a job. """
        all_receive_times = self.last_receive_times.values()
        return max(all_receive_times)

    def _worker_is_free(self, wid):
        """ Return True if worker wid is free """
        if wid in self.free_workers:
            return True
        worker_result_file_name = self._get_result_file_name_for_worker(wid)
        if os.path.exists(worker_result_file_name):
            self._read_result_from_worker_and_update(wid)
        else:
            return False

    def a_worker_is_free(self):
        """ Return wid if any worker is free """
        for wid in self.worker_ids:
            if self._worker_is_free(wid):
#                print('worker ',wid,' is free')
                return self._get_last_receive_time()
        return None


    def all_workers_are_free(self):
        """ return True if all workers are free """
        all_free = True
        for wid in self.worker_ids:
            all_free = (all_free and self._worker_is_free(wid))

        if all_free:
            return self._get_last_receive_time()
        else:
            return None

    def _dispatch_evaluation(self, func_caller, point_dict, qinfo, worker_id, **kwargs):
        """ dispatches evaluation to worker_id """
        '''
        args:
            point_dict : dictionary {'X' : all X's searched so far, 'Y': all Y's sensed so far}
            func_caller : the function ThompsonActiveSearch in file TS.py
            worker_id : the agent that will perform this computation
            qinfo :
        '''
        if self.qinfos_in_progress[worker_id] is not None:
            err_msg = 'qinfos_in_progress: %s,\nfree_workers: %s.'%(
                        str(self.qinfos_in_progress), str(self.free_workers))
#            print(err_msg)
            raise ValueError('Check if worker is free before sending evaluation.')
        # First add all the data to qinfo
        qinfo.worker_id = worker_id
        qinfo.working_dir = self.working_dir_names[worker_id]
        qinfo.result_file = self._get_result_file_name_for_worker(worker_id)
        qinfo.point = point_dict
        # Create the working directory
        os.makedirs(qinfo.working_dir)
        # Dispatch the evaluation in a new process
        target_func = lambda: func_caller.ActiveSearch(point_dict, qinfo)# <- calls ThompsonActiveSearch(point_dict) , qinfo, **kwargs)
#        print("dispatching agent: ",worker_id)
#        print("free agents: ",self.free_workers)
        self.worker_processes[worker_id] = Process(target=target_func)
        self.worker_processes[worker_id].start()
        # Add the qinfo to the in progress bar and remove from free_workers
        self.qinfos_in_progress[worker_id] = qinfo
        self.free_workers.discard(worker_id)


    def dispatch_single_evaluation(self, func_caller, point, qinfo, **kwargs):
        """ Dispatches a single evaluation to a free worker """
        wid = self.free_workers.pop()
        self._dispatch_evaluation(func_caller, point, qinfo, wid, **kwargs)


    def dispatch_batch_of_evaluations(self, func_caller, points, qinfos, **kwargs):
        """ Dispatches a batch of evaluations; number of evaluation points == number of workers available in total"""

        # assert len(points['X']) == self.num_workers
        # assert len(points['Y']) == self.num_workers
        assert (len(points['X'])%self.num_workers) == 0
        for wid in range(self.num_workers):
            self._dispatch_evaluation(func_caller, points, qinfos[wid], self.worker_ids[wid], **kwargs)
