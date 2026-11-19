"""Saved-state analysis for the continuing/reversing reserve comparison."""
import json
import shutil

import numpy as np
import torch

from scripts.compare_food_direction import NAMES, displacement
from scripts.analyze_food_movement import initial_equality, rows as numeric_rows
from scripts.compare_trail_reference import load_state, load_resource, headless_image, require, read_json, save_json, sha256, write_csv
from scripts.demo_controller_lookup_fix import event_analysis, read_rows
from PIL import Image


def analyze(output, configs, reference_dir):
    local=output/'local'
    runs={n:local/n/'run' for n in NAMES}
    previous_run=reference_dir/'local/faster/run'
    validation=initial_equality({**runs,'previous_faster':previous_run})
    (output/'measurements').mkdir(exist_ok=True)
    logs,events={},{}
    for name in NAMES:
        for f in ('resource_metrics.csv','cell_events.csv'):
            shutil.copy2(runs[name]/f,output/'measurements'/f'{name}_{f}')
        logs[name]={int(r['completed_update']):r for r in numeric_rows(runs[name]/'resource_metrics.csv')}
        require(list(logs[name])==list(range(1001)),'Incomplete resource log')
        events[name]=read_rows(runs[name]/'cell_events.csv')
    totals,_=event_analysis(local,configs)
    save_json(output/'event_summary.json',totals)
    initial_resource=load_resource(runs['continues'],0)
    initial_capacity=initial_resource['capacity']
    original_source=initial_capacity>0
    turn_source=load_resource(runs['continues'],500)['capacity']>0
    expected=torch.from_numpy(initial_capacity.copy())
    requested=0.0
    for u in range(1001):
        if u:
            injection=expected.clamp(max=.02)
            requested+=float(injection.double().sum())
            expected.sub_(injection).clamp_(min=0)
        for n in NAMES:
            r=logs[n][u]
            require(abs(r['cumulative_requested']-requested)<1e-7,'Release differs from independent accounting')
            require(abs(r['remaining_capacity']-float(expected.double().sum()))<1e-7,'Reserve loss/creation')
            require(abs(r['cumulative_requested']-r['cumulative_delivered']-r['cumulative_clipped'])<.02,'Unexplained actual-delivery discrepancy')
    previous_logs={int(r['completed_update']):r for r in numeric_rows(previous_run/'resource_metrics.csv')}
    repeat_log_differences=[{'completed_update':u,**{k:logs['continues'][u][k]-previous_logs[u][k] for k in previous_logs[u] if k!='completed_update'}} for u in range(1001)]
    write_csv(output/'repeat_resource_log_differences.csv',repeat_log_differences)
    prefix_log_differences={k:max(abs(logs['continues'][u][k]-logs['reverses'][u][k]) for u in range(625)) for k in logs['continues'][0]}
    prefix_events_equal=events['continues'][:624]==events['reverses'][:624]
    updates=list(range(0,1001,25))
    snapshots,pairs,regions,region_changes,spatial_changes,repeats=[],[],[],[],[],[]
    previous_occ,previous_masks={},{}
    expected=torch.from_numpy(initial_capacity.copy())
    last=0
    for u in updates:
        for _ in range(last,u):expected.sub_(expected.clamp(max=.02)).clamp_(min=0)
        states,resources,occupied={},{},{}
        for n in NAMES:
            state=states[n]=load_state(runs[n],u)
            resource=resources[n]=load_resource(runs[n],u)
            require(tuple(state.shape)==(1,14,400,400) and bool(torch.isfinite(state).all()),'Invalid captured state')
            require(bool((state[0,13]==state[0,13].floor()).all()),'Invalid genome index')
            occupied[n]=state[0,13].numpy()>=0
            capacity=resource['capacity']
            offset=displacement(n,u)
            require(np.array_equal(capacity,np.roll(expected.numpy(),offset,axis=1)),'Source mask/translation phase differs')
            ys=np.where(capacity>0)[1]
            require(int(ys.min())>=0 and int(ys.max())<400,'Source outside supported domain')
            require(int(ys.min())>=18 and int(ys.max())<=398,'Source wrapped across boundary')
            r=logs[n][u]
            require(r['occupied_cells']==int(occupied[n].sum()),'Occupancy capture phase differs')
            require(r['occupied_inside_source']==int((occupied[n]&(capacity>0)).sum()),'Source-region capture phase differs')
            require(abs(r['total_energy']-float(state[0,0].double().sum()))<1e-7,'Energy capture phase differs')
            require(abs(r['total_infra']-float(state[0,1].double().sum()))<1e-7,'Infrastructure capture phase differs')
            with Image.open(runs[n]/'screenshots'/f'step_{u:07d}.png') as image:
                require(np.array_equal(np.asarray(image),np.asarray(headless_image(state))),'Saved RGB/capture mismatch')
            if u==1000:
                require(torch.equal(state,torch.load(runs[n]/'checkpoint_0001000/substrate.pt',weights_only=True)),'Final checkpoint differs')
            snapshots.append({'condition':n,**r,'source_displacement_y_cells':offset,
                'state_sha256':sha256(runs[n]/'frame_states'/f'update_{u:07d}.npz'),
                'resource_sha256':sha256(runs[n]/'frame_states'/f'resource_{u:07d}.npz')})
            prev=previous_occ.get(n,occupied[n])
            spatial_changes.append({'condition':n,'previous_capture_update':last,'completed_update':u,
                'empty_to_occupied_endpoint_transitions':int((~prev&occupied[n]).sum()),
                'occupied_to_empty_endpoint_transitions':int((prev&~occupied[n]).sum()),
                'retained_occupied_endpoints':int((prev&occupied[n]).sum())})
        pairs.append({'completed_update':u,'differing_occupied_locations':int((occupied['continues']!=occupied['reverses']).sum()),
            'maximum_abs_state_difference':float((states['continues']-states['reverses']).abs().max()),
            'reverses_minus_continues_occupied':int(occupied['reverses'].sum())-int(occupied['continues'].sum())})
        masks={'original_source':original_source,'turn_source_at_500':turn_source,
            'continues_current_source':resources['continues']['capacity']>0,
            'reverses_current_source':resources['reverses']['capacity']>0}
        for label,mask in masks.items():
            counts={n:int((occupied[n]&mask).sum()) for n in NAMES}
            regions.append({'completed_update':u,'region':label,'source_cells':int(mask.sum()),
                'continues_occupied':counts['continues'],'reverses_occupied':counts['reverses'],
                'reverses_minus_continues':counts['reverses']-counts['continues']})
            for n in NAMES:
                prev=previous_occ.get(n,occupied[n])
                prior_mask=previous_masks.get(label,mask)
                already=int((prev&mask).sum())
                gains=int((~prev&occupied[n]&mask).sum())
                losses=int((prev&~occupied[n]&mask).sum())
                crossing=already-int((prev&prior_mask).sum())
                require(counts[n]-int((prev&prior_mask).sum())==gains-losses+crossing,'Mask/endpoint decomposition failed')
                region_changes.append({'condition':n,'region':label,'previous_capture_update':last,'completed_update':u,
                    'occupied_in_current_region':counts[n],'previously_occupied_in_current_region':already,
                    'empty_to_occupied_endpoint_transitions_in_current_region':gains,
                    'occupied_to_empty_endpoint_transitions_in_current_region':losses,
                    'mask_crossing_preexisting_occupied_contribution':crossing})
        old=load_state(previous_run,u)
        old_resource=load_resource(previous_run,u)
        repeats.append({'completed_update':u,'differing_occupied_locations':int(((old[0,13].numpy()>=0)!=occupied['continues']).sum()),
            'maximum_abs_state_difference':float((old-states['continues']).abs().max()),
            'reserve_fields_bitwise_equal':np.array_equal(old_resource['capacity'],resources['continues']['capacity']),
            'cumulative_delivery_difference':resources['continues']['cumulative_delivered']-old_resource['cumulative_delivered']})
        previous_occ={n:o.copy() for n,o in occupied.items()}
        previous_masks={n:m.copy() for n,m in masks.items()}
        last=u
    write_csv(output/'snapshot_measurements.csv',snapshots)
    write_csv(output/'paired_measurements.csv',pairs)
    write_csv(output/'spatial_changes.csv',spatial_changes)
    write_csv(output/'source_region_counts.csv',regions)
    write_csv(output/'source_region_changes.csv',region_changes)
    write_csv(output/'repeat_comparison.csv',repeats)
    windows=[]
    for n in NAMES:
        for label,lo,hi in (('common_schedule',1,624),('divergent_schedule',625,1000),('first_post_turn_capture_interval',625,650),('total',1,1000)):
            selected=events[n][lo-1:hi]
            windows.append({'condition':n,'window':label,'first_update':lo,'last_update':hi,
                **{k:sum(r[k] for r in selected) for k in selected[0] if k.endswith(('_gains','_losses','_replacements'))}})
    write_csv(output/'phase_event_totals.csv',windows)
    summary,budgets=[],{}
    for n in NAMES:
        r=logs[n][1000]
        b=read_json(runs[n]/'meta.json')['resource_budget']
        b['reserve_depletion_minus_requested']=b['initial_capacity']-b['remaining_capacity']-b['cumulative_requested']
        b['requested_minus_delivered']=b['cumulative_requested']-b['cumulative_delivered']
        budgets[n]=b
        initial_row=next(x for x in regions if x['completed_update']==1000 and x['region']=='original_source')
        returning=next(x for x in regions if x['completed_update']==1000 and x['region']=='reverses_current_source')
        summary.append({'condition':n,'initial_occupied':int(logs[n][0]['occupied_cells']),
            'final_occupied':int(r['occupied_cells']),'final_occupied_inside_own_source':int(r['occupied_inside_source']),
            'final_occupied_in_original_region':initial_row[n+'_occupied'],
            'final_occupied_in_same_returning_region':returning[n+'_occupied'],
            'local_colonization_gains':totals[n]['colonization_gains'],'radiation_reseeding_gains':totals[n]['radiation_gains'],
            'actual_delivered_energy':r['cumulative_delivered'],'requested_energy':r['cumulative_requested'],
            'remaining_reserve':r['remaining_capacity'],'clipped_energy':r['cumulative_clipped']})
    write_csv(output/'comparison.csv',summary)
    delivery_range=max(abs(logs['continues'][u]['cumulative_delivered']-logs['reverses'][u]['cumulative_delivered']) for u in range(1001))
    save_json(output/'resource_accounting.json',{'runs':budgets,'maximum_cumulative_actual_delivery_difference':delivery_range,
        'fraction_of_final_requested_energy':delivery_range/logs['continues'][1000]['cumulative_requested'],
        'units':'Simulator energy units. Actual delivered = measured after-minus-before addition/clamp each update.',
        'residual':'Repeated float32 reserve subtraction has a nonzero depletion-minus-request residual; independently checked and retained, without rescaling.'})
    prefix=[r for r in pairs if r['completed_update']<625]
    validation.update({'resource_budget_checks':2002,'state_rgb_source_phase_checks':82,'final_checkpoint_matches':2,
        'source_region_pair_counts':len(regions),'mask_endpoint_decomposition_checks':len(region_changes),
        'phase_event_update_checks':2000,'resource_wraps':0,
        'captured_common_prefix_max_abs_state_difference':max(r['maximum_abs_state_difference'] for r in prefix),
        'captured_common_prefix_max_occupancy_difference':max(r['differing_occupied_locations'] for r in prefix),
        'last_common_schedule_capture_update':600,'first_different_source_shift_completed_update':625,
        'common_schedule_resource_log_max_abs_differences_through_624':prefix_log_differences,
        'common_schedule_event_rows_through_624_equal':prefix_events_equal,
        'first_per_update_total_occupancy_difference':next((u for u in range(1001) if logs['continues'][u]['occupied_cells']!=logs['reverses'][u]['occupied_cells']),None),
        'first_different_event_count_row':next(({'completed_update':a['completed_update'],'different_fields':[k for k in a if a[k]!=b[k]]} for a,b in zip(events['continues'],events['reverses']) if a!=b),None),
        'radiation_gains_updates_625_through_650':{n:sum(r['radiation_gains'] for r in events[n][624:650]) for n in NAMES},
        'first_captured_occupancy_difference':next((r['completed_update'] for r in pairs if r['differing_occupied_locations']),None),
        'previous_faster_repeat_max_abs_state_difference':max(r['maximum_abs_state_difference'] for r in repeats),
        'previous_faster_repeat_max_occupancy_difference':max(r['differing_occupied_locations'] for r in repeats),
        'previous_faster_repeat_max_actual_delivery_difference':max(abs(r['cumulative_delivery_difference']) for r in repeats),
        'previous_faster_repeat_all_reserve_fields_equal':all(r['reserve_fields_bitwise_equal'] for r in repeats),
        'previous_faster_repeat_per_update_resource_max_abs_differences':{k:max(abs(r[k]) for r in repeat_log_differences) for k in repeat_log_differences[0] if k!='completed_update'}})
    save_json(output/'validation.json',validation)
    plot(output,logs,pairs,regions)
    print(json.dumps({'comparison':summary,'validation':validation,'delivery_range':delivery_range},indent=2),flush=True)


def plot(output,logs,pairs,regions):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(10,9),constrained_layout=True)
    for n in NAMES:
        axes[0].plot(list(logs[n]),[r['occupied_cells'] for r in logs[n].values()],label=n)
        selected=[r for r in regions if r['region']=='reverses_current_source']
        axes[1].plot([r['completed_update'] for r in selected],[r[n+'_occupied'] for r in selected],label=n+' in same returning mask')
    axes[0].set(ylabel='Occupied cells',title='Matched start; direction schedules first differ at update 625')
    axes[1].set(ylabel='Occupied cells in source region')
    axes[2].plot([r['completed_update'] for r in pairs],[r['differing_occupied_locations'] for r in pairs])
    axes[2].set(ylabel='Different occupied locations',xlabel='Completed update')
    for ax in axes:
        ax.axvline(500,color='gray',ls=':',lw=1)
        ax.axvline(625,color='black',ls='--',lw=1)
        ax.grid(alpha=.2)
    axes[0].legend();axes[1].legend()
    fig.savefig(output/'trajectories.png',dpi=140)
    plt.close(fig)
