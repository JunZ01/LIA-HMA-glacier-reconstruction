from pathlib import Path
src=Path(__file__).with_name('yx_add_thickness_change_rasters_v1.py')
text=src.read_text(encoding='utf-8')
old="out=Path(json.loads((BASE/'latest_completed_reconstruction.json').read_text(encoding='utf-8'))['path']);volumes="
new="assert len(sys.argv)==2,'Provide the new immutable delivery directory';out=Path(sys.argv[1]);assert not (out/'thickness_change_raster_QC.csv').exists(),'Do not overwrite completed change rasters';volumes="
assert text.count(old)==1
text=text.replace(old,new)
text=text.replace("'composition_label_note':'For version12", "'composition_label_note':'The full scientific family name is authoritative; version109 is the incremental compact-S forest,110 is the exact PP kernel. For version12")
dst=src.with_name('yx_add_thickness_change_rasters_v2.py')
assert not dst.exists()
dst.write_text(text,encoding='utf-8')
print(str(dst))
