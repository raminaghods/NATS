'''
Code for submitted work:

``Multi Agent Active Search using Realistic Depth-Aware Noise Model'',
submitted to 2020 Conference on Robot Learning
author: anonymous

please do not distribute. The code will become public upon acceptance of the paper.

Parallelized implementation of bayesian optimization for unknown function.

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

from argparse import Namespace
import time
import numpy as np
import pickle as pkl

# def get_option_specs(name, required=False, default=None, help_str='', **kwargs):
#     """ A wrapper function to get a specification as a dictionary. """
#     ret = {'name':name, 'required':required, 'default':default, 'help':help_str}
#     for key, value in kwargs.items():
#         ret[key] = value
#     return ret

# optimizer_args = [
#   get_option_specs('max_num_steps', False, 1e7,
#     'If exceeds this many evaluations, stop.'),
#   get_option_specs('num_init_evals', False, 0,
#     ('The amount of evaluations for initialisation. If <0, will use default.')),
#   get_option_specs('num_workers', False, 1,
#     'The number of workers in parallel.'),
#   get_option_specs('mode', False, 'asy',
#     'If \'syn\', uses synchronous parallelisation, else asynchronous.'),
#   get_option_specs('build_new_model_every', False, 27,
#     'Updates the GP via a suitable procedure every this many iterations.'),
#   get_option_specs('report_results_every', False, 20,
#     'Report results every this many iterations.'),
#   ]



class Bayesian_optimizer(object):
    """ class co-ordinates the optimization process"""

    def __init__(self, worker_manager, func_caller, options=None):
        """ Constructor """
        self.worker_manager = worker_manager
        self.func_caller = func_caller
        self.options = options

        self.X = None
        self.Y = None
        self.full_recovery_rate = []
        self.partial_recovery_rate = []
        self.par = None
#        self.gamma = None
#        self.B = None
        self.num_points = []
        # self.beta_hats = []

        self._set_up()


    def _set_up(self):
        """ sets up the optimizer """
        self.available_capital = 0.0
        self.step = 0
        self.num_completed_evals = 0
        self.step_idx = 0
        # Initialise step idx
        self.curr_opt_val = -np.inf
        # self.curr_opt_pt = None
        # self.curr_true_opt_val = -np.inf
        # self.curr_true_opt_pt = None

        # Initialise worker manager
        self.worker_manager.set_optimiser(self)
        copyable_params_from_worker_manager = ['num_workers']
        for param in copyable_params_from_worker_manager:
          setattr(self, param, getattr(self.worker_manager, param))
        # Other book keeping stuff
        self.last_report_at = 0
        #self.last_model_build_at = 0
        self.eval_points_in_progress = []
        self.eval_idxs_in_progress = []
        # Set initial history
        # query infos will maintain a list of namespaces which contain information about
        # the query in send order. Everything else will be saved in receive order.
        self.history = Namespace(query_step_idxs=np.zeros(0),
                                 # query_points=np.zeros((0, self.domain_dim)),
                                 query_vals=np.zeros(0),
                                 # query_true_vals=np.zeros(0),
                                 query_send_times=np.zeros(0),
                                 query_receive_times=np.zeros(0),
                                 query_eval_times=np.zeros(0),
                                 curr_opt_vals=np.zeros(0),
                                 #curr_true_opt_vals=np.zeros(0),
                                 query_infos=[],
                                 worker_info={k:[] for k in self.worker_manager.worker_ids},
                                )
        # Finally call the child set up.
        #self._child_set_up()
        # Post child set up.
        # method_prefix = 'asy' if self.is_asynchronous() else 'syn'
        # self.full_method_name = method_prefix + '-' + self.method_name
        # Set pre_eval_points and vals
        # self.pre_eval_points = np.zeros((0, self.domain_dim))
        # self.pre_eval_vals = np.zeros(0)
        # self.pre_eval_true_vals = np.zeros(0)


    def add_capital(self, capital):
        """ here capital is the max number of time steps we want to run our optimization """
        self.available_capital += float(capital)

    def initialise_capital(self):
        self.spent_capital = 0.0

    def get_curr_spent_capital(self):
        return self.step_idx #self.spent_capital

    def set_curr_spent_capital(self, value):
        self.spent_capital = float(value)

    def perform_initial_queries(self):
        if self.options.num_init_evals <= 0:
            num_init_evals = 0.
        else:
            num_init_evals = int(self.options.num_init_evals)
        num_init_evals = max(self.num_workers, num_init_evals)

        #self.sample_from_prior(num_init_evals) #TODO : sample X, Y from prior (for TS)

        self.pre_eval_betas = self.options.GP.sample_from_prior(num_init_evals)
        for idx in range(num_init_evals):
            self.step_idx += 1
            self._wait_for_a_free_worker()
            self._dispatch_single_evaluation_to_worker_manager({'beta':self.pre_eval_betas[idx]}, pre_eval=True)#{'X' : X[idx], 'Y' : Y[idx]}) #TODO : modify signature of function to accept a dict of X,Y


    def _wait_till_free(self, is_free, poll_time):
        """ Waits until is_free returns true. """
        keep_looping = True
        while keep_looping:
            # print('in here!')
            last_receive_time = is_free()
            if last_receive_time is not None: #implies a worker is free and we obtained the last receive time from a worker
            # Get the latest set of results and dispatch the next job.
                self.set_curr_spent_capital(last_receive_time)
                latest_results = self.worker_manager.fetch_latest_results()
                self._add_data_to_model(latest_results)
                for qinfo_result in latest_results:
                    self._update_history(qinfo_result)
                    self._remove_from_in_progress(qinfo_result)
                keep_looping = False
            else:
                time.sleep(poll_time)

    def _wait_for_a_free_worker(self):
        """waits for a free worker and updates """
        self._wait_till_free(self.worker_manager.a_worker_is_free, self.worker_manager.poll_time)

    def _wait_for_all_free_workers(self):
        self._wait_till_free(self.worker_manager.all_workers_are_free, self.worker_manager.poll_time)


    def _add_data_to_model(self, qinfos):
        if len(qinfos) == 0:
            return
        n = len(qinfos)
        for i in range(n):
            x = qinfos[i].val['x']
            y = qinfos[i].val['y']
            # self.full_recovery_rate.append(qinfos[i].val['full_recovery_rate'])
            # self.partial_recovery_rate.append(qinfos[i].val['partial_recovery_rate'])
            self.par = qinfos[i].val['par']
            # self.beta_hats.append(self.par[0])
            if 'pre-eval' in list(qinfos[i].val.keys()):
                # assert len(y) == 1
                # assert self.X == None
                # print('X: ',x)
                # print('Y: ',y)
                self.par_init = qinfos[i].val['par']
                for idx, xx in enumerate(y):
                    if self.X is None:
                        self.X = x[idx]#np.array([x[idx]])
                        self.Y = y[idx]#y[idx]
                        self.num_points.append(y[idx].shape[0])
                        # assert self.X.shape == (1,128)
                    else:
                        self.X = np.append(self.X, x[idx], axis=0)
                        self.Y = np.append(self.Y, y[idx], axis=0)#.reshape((-1,1))
                        self.num_points.append(y[idx].shape[0])

                        # assert self.Y.shape[1] == 1
                        #assert self.X.shape[2] == 128
            else:
                self.X = np.append(self.X, x, axis=0)
                self.Y = np.append(self.Y, y, axis=0)
                self.num_points.append(y.shape[0])


    #TODO : didn't change this function yet - understand what qinfos is and modify accordingly
    def _update_history(self, qinfo):
        """ Data is a namespace which contains a lot of ancillary information. val
            is the function value. """
        # First update the optimal point and value.
        # if qinfo.val > self.curr_opt_val:
        #   self.curr_opt_val = qinfo.val
        #   self.curr_opt_pt = qinfo.point
        # if hasattr(qinfo, 'true_val') and qinfo.true_val > self.curr_true_opt_val:
        #   self.curr_true_opt_val = qinfo.true_val
        #   self.curr_true_opt_pt = qinfo.point
        # Now store in history
        self.history.query_step_idxs = np.append(self.history.query_step_idxs,
                                                 qinfo.step_idx)
        # self.history.query_points = np.append(self.history.query_points,
        #   qinfo.point.reshape((-1, self.domain_dim)), axis=0)
        self.history.query_vals = np.append(self.history.query_vals, qinfo.val)
        # self.history.query_true_vals = np.append(self.history.query_true_vals, qinfo.true_val)
        self.history.query_send_times = np.append(self.history.query_send_times,
                                                  qinfo.send_time)
        self.history.query_receive_times = np.append(self.history.query_receive_times,
                                                     qinfo.receive_time)
        self.history.query_eval_times = np.append(self.history.query_eval_times,
                                                     qinfo.eval_time)
        self.history.curr_opt_vals = np.append(self.history.curr_opt_vals, self.curr_opt_val)
        # self.history.curr_true_opt_vals = np.append(self.history.curr_true_opt_vals,
                                                    # self.curr_true_opt_val)
        self.history.worker_info[qinfo.worker_id].append(qinfo.step_idx)
        self.history.query_infos.append(qinfo)


    def _remove_from_in_progress(self, qinfo):
        """ Removes a job from the in progress status. """
        completed_eval_idx = self.eval_idxs_in_progress.index(qinfo.step_idx)
        self.eval_idxs_in_progress.pop(completed_eval_idx)
        self.eval_points_in_progress.pop(completed_eval_idx)


    def _add_to_in_progress(self, qinfos):
        for qinfo in qinfos:
            self.eval_idxs_in_progress.append(qinfo.step_idx)
            self.eval_points_in_progress.append(qinfo.point)


    def _dispatch_single_evaluation_to_worker_manager(self, point_dict, **kwargs):
        """ for asynchronous """
        # print("step_idx: ",self.step_idx," send_time: ",self.get_curr_spent_capital())

        qinfo = Namespace(send_time=self.get_curr_spent_capital(), step_idx=self.step_idx,
                      point_dict=point_dict)
        if kwargs['pre_eval']:
            qinfo.compute_posterior = False
        else:
            qinfo.compute_posterior = True

        self.worker_manager.dispatch_single_evaluation(self.func_caller, point_dict, qinfo) #TODO : check this function signature - point_dict, qinfo / just point, qinfo ?
        self._add_to_in_progress([qinfo])

    def _dispatch_batch_of_evaluations_to_worker_manager(self, point_dict, **kwargs):
        """ for synchronous """
        if kwargs['pre_eval']:
            qinfos = [Namespace(send_time=self.get_curr_spent_capital()+j, step_idx=self.step_idx+j,
                point_dict=point_dict, compute_posterior=False) for j in range(self.num_workers)]
        else:
            qinfos = [Namespace(send_time=self.get_curr_spent_capital()+j, step_idx=self.step_idx+j,
                point_dict=point_dict, compute_posterior=True) for j in range(self.num_workers)]
        self.worker_manager.dispatch_batch_of_evaluations(self.func_caller, point_dict, qinfos)
        self._add_to_in_progress(qinfos)

        # raise NotImplementedError('synchronous not implemented yet.')


    #def _child_init(self):



    def optimise_initialise(self):
        # self.curr_opt_pt = None
        self.curr_opt_val = -np.inf
        self.initialise_capital()
        self.perform_initial_queries()
        #self._child_init()


    def _terminate_now(self):
        """ return True if we should terminate now """
        if(self.options.check_performance and self.X is not None):
            _, _, _, beta_hat, _ = self.options.GP.getPosterior(self.X,self.Y,self.par,self.get_curr_spent_capital())
            est = (np.round(beta_hat)>(np.amax(beta_hat)/2))
            real = (self.func_caller.beta>0)
            if(np.all(est==real)):
                return True
        if self.step_idx >= self.options.max_num_steps:
            return True
        return self.get_curr_spent_capital() >= self.available_capital


    def is_asynchronous(self):
        return self.options.mode.lower().startswith('asy')


    def _asynchronous_optimise_routine(self): #TODO
        self._wait_for_a_free_worker()
        #next_pt = self._determine_next_eval_point() #TODO :  push this part into TS routine

        self.step_idx += 1

        self._dispatch_single_evaluation_to_worker_manager({'X':self.X, 'Y':self.Y, 'par':self.par}, pre_eval=False)
        # self.step_idx += 1


    def _synchronous_optimise_routine(self): #TODO
        self._wait_for_all_free_workers()
        # next_batch_of_points = self._determine_next_batch_of_eval_points() #TODO :  push this part into TS routine

        self.step_idx += self.num_workers

        self._dispatch_batch_of_evaluations_to_worker_manager({'X':self.X, 'Y':self.Y, 'par':self.par}, pre_eval=False)
        # self.step_idx += self.num_workers
        # raise NotImplementedError('synchronous not implemented yet.')


    def _report_curr_results(self):
        raise NotImplementedError('not implemented')


    def _optimise_wrap_up(self):
        self.worker_manager.close_all_jobs() #TODO
        self._wait_for_all_free_workers()
        # self._report_curr_results() #TODO
        # Store additional data
        self.beta_hats = []
        index = 0
        for i in range(len(self.num_points)):
            print('returning results',i)
            index = index + self.num_points[i]
            _, _, _, beta_hat, self.par_init = self.options.GP.getPosterior(self.X[:(index)],self.Y[:(index)],self.par_init,(i+1))
            self.beta_hats.append(beta_hat)
        if(self.options.check_performance):
            for ii in range(int(self.available_capital-self.get_curr_spent_capital())):
                self.beta_hats.append(beta_hat)

        _, _, _, self.curr_opt_val, _ = self.options.GP.getPosterior(self.X, self.Y, self.par, self.get_curr_spent_capital())
        # self.history.num_jobs_per_worker = np.array(self._get_jobs_for_each_worker()) #TODO
        # self.history.full_method_name = self.full_method_name #TODO


    def optimize(self, max_capital):
        """ performs the optimization routine """

        self.add_capital(max_capital)
        self.optimise_initialise()

        # Main Loop --------------------------
        while not self._terminate_now():
            if self.is_asynchronous():
                self._asynchronous_optimise_routine()
            else:
                self._synchronous_optimise_routine()

          # if self.step_idx - self.last_model_build_at >= self.options.build_new_model_every:
          #   self._build_new_model()
          # if self.step_idx - self.last_report_at >= self.options.report_results_every:
          #   self._report_curr_results()


        # Wrap up and return
        self._optimise_wrap_up()
        return self.beta_hats#self.curr_opt_val, self.history, self.full_recovery_rate, self.partial_recovery_rate
