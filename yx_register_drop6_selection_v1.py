from pathlib import Path
import json

# Reuse the validated transaction, changing only this new derivative's date and authorization.
namespace = {'__file__': str(Path(__file__).resolve())}
source = Path('yx_followup_transaction_v1.py').read_text(encoding='utf-8')
source = source.replace('_20261009_v1', '_20261010_v1')
source = source.replace("user_authorization='完全完成; 并行完成'", "user_authorization='2026-10-10 explicit user request: 丢弃这6条冰川'")
exec(compile(source, 'yx_selection_transaction_v1', 'exec'), namespace)
register = namespace['register']
EXCLUDED = ['YX-03055','YX-03621','YX-05103','YX-05176','YX-14077','YX-28236']
OUT = Path('H:/ToYangxin/result_yx/selected_reconstruction_20261010_v1_654')
print(register(
    'yx_selected_sample654_drop6_v1',
    {'status':'user-authorized-six-system-exclusion-derivative-preparation',
     'source_delivery':'H:/ToYangxin/result_yx/completed_reconstruction_20261009_v4_654',
     'output_directory':str(OUT),'excluded_systems':EXCLUDED,
     'excluded_source_records':EXCLUDED+['YX-29305'],
     'original_system_population':660,'analysis_system_population':654,
     'original_source_records':662,'analysis_source_records':655,
     'original_raw_inventory_modified':False,'original_deliveries_modified':False,
     'model_recalculations_authorized':False,'formal191_count_effect':0},
    'YX analysis sample: user excludes six held systems from all subsequent sample statistics',
    'The user explicitly instructs on2026-10-10 to discard the six held glacier systems03055/03621/05103/05176/14077/28236 from the analysis sample. Interpret as a disclosed sample exclusion in a new immutable derivative, preserving original inventory,660-system assessed population and verified654-model delivery. Joint05176 also removes source29305, hence seven original source records are excluded:662->655 source records and660->654 systems. All subsequent selected-sample area/modern reference/mountain/GIS summaries must use the same654 systems. Previously qualified654 endpoint models and modeled volume totals are unchanged and copied/referenced without recomputation. Record the six exclusions and their prior scientific reasons, do not claim they reconstructed or erase prior failure evidence. Recalculate selected area/native modern baseline and10-local-mountain counts; preserve no-sample mountain blanks and native thickness support limitations. Prepare new selected654 GIS/Shapefile and report with explicit upstream660-to654 selection. No simulation or mainline191 changes authorized.',
    False, [],
    'Build and independently verify a selected654 derivative with655 source records and exclusion provenance, then publish its current pointer; no models or original files modified.',
    phase='selection-authorized'))
