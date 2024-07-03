from copy import deepcopy

result_dict = {'TaskSpecific': {'acc_this_step': -1.0},
               'TaskAgnostic': {'acc_prev_w_cluster': -1.0,
                                'acc_prev_wo_cluster': -1.0,
                                'acc_all_w_cluster': -1.0,
                                'acc_all_wo_cluster': -1.0,
                                'acc_stepwise_w_cluster': {},
                                'acc_stepwise_wo_cluster': {},
                                }}

class StepResults:
    def __init__(self, num_steps):
        self.step_results_list = [deepcopy(result_dict) for s in range(num_steps)]

    def update_step(self, step,
                    acc_single_head_this_step_w_cluster,
                    acc_all_prev_test_w_cluster, acc_all_prev_test_wo_cluster,
                    acc_all_test_w_cluster, acc_all_test_wo_cluster,
                    acc_step_test_w_cluster_dict, acc_step_test_wo_cluster_dict):

        self.step_results_list[step]['TaskSpecific'] = acc_single_head_this_step_w_cluster
        self.step_results_list[step]['TaskAgnostic']['acc_prev_w_cluster'] = acc_all_prev_test_w_cluster
        self.step_results_list[step]['TaskAgnostic']['acc_prev_wo_cluster'] = acc_all_prev_test_wo_cluster
        self.step_results_list[step]['TaskAgnostic']['acc_all_w_cluster'] = acc_all_test_w_cluster
        self.step_results_list[step]['TaskAgnostic']['acc_all_wo_cluster'] = acc_all_test_wo_cluster
        self.step_results_list[step]['TaskAgnostic']['acc_stepwise_w_cluster'] = acc_step_test_w_cluster_dict
        self.step_results_list[step]['TaskAgnostic']['acc_stepwise_wo_cluster'] = acc_step_test_wo_cluster_dict

    def show_step(self, step=0):
        print('\n========================================================')
        print(f'         {step} Stage Final Test Output (test split)             ')
        print(f'[S{step}-Single Head]')
        print(f"Acc_this_step             = {self.step_results_list[step]['TaskSpecific']}")

        print(f'\n[S{step}-Joint Head]')
        print('All-Previous-Discovered-Test')
        print(f"Acc_all_prev_W_cluster    = {self.step_results_list[step]['TaskAgnostic']['acc_prev_w_cluster']}")
        print(f"Acc_all_prev_WO_cluster   = {self.step_results_list[step]['TaskAgnostic']['acc_prev_wo_cluster']}")

        print('\nAll-Discovered-Test')
        print(f"Acc_all_W_cluster         = {self.step_results_list[step]['TaskAgnostic']['acc_all_w_cluster']}")
        print(f"Acc_all_WO_cluster        = {self.step_results_list[step]['TaskAgnostic']['acc_all_wo_cluster']}")

        print('\nStepwise-Discovered')
        print('Step Single Test w/ clustering dict')
        print(self.step_results_list[step]['TaskAgnostic']['acc_stepwise_w_cluster'])

        print('Step Single Test w/o clustering dict')
        print(self.step_results_list[step]['TaskAgnostic']['acc_stepwise_wo_cluster'])
        print('========================================================')

        return self.step_results_list[step]

    def return_step(self, step):
        return self.step_results_list[step]

