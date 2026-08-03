from __future__ import annotations
from pathlib import Path, PurePosixPath
import base64, io, zipfile, shutil
from scalebridge.integration.energyplus.prototypes import resolve_generated_data_root, sha256_file
from scalebridge.integration.energyplus.generation.campaign_definition import SourceFileRef
MAX_FILES=256; MAX_UNCOMPRESSED=2*1024*1024*1024

def import_zip(contents, campaign_id):
    if not contents or ',' not in contents: raise ValueError('No ZIP payload was supplied')
    _,encoded=contents.split(',',1); raw=base64.b64decode(encoded); z=zipfile.ZipFile(io.BytesIO(raw))
    infos=[i for i in z.infolist() if not i.is_dir()]
    if len(infos)>MAX_FILES or sum(i.file_size for i in infos)>MAX_UNCOMPRESSED: raise ValueError('ZIP exceeds import limits')
    accepted=[]
    for info in infos:
        p=PurePosixPath(info.filename)
        if p.is_absolute() or '..' in p.parts: raise ValueError(f'Unsafe ZIP path: {info.filename}')
        parts=[x for x in p.parts if x not in ('','.')] 
        if len(parts)!=2 or parts[0].casefold() not in {'idf','epw'}: raise ValueError('ZIP must contain only idf/*.idf and epw/*.epw')
        ext='.'+parts[0].casefold()
        if p.suffix.casefold()!=ext: raise ValueError(f'Unexpected file type: {info.filename}')
        accepted.append((info,parts[0].casefold(),parts[1]))
    if not any(k=='idf' for _,k,_ in accepted) or not any(k=='epw' for _,k,_ in accepted): raise ValueError('ZIP requires at least one IDF and one EPW')
    root=resolve_generated_data_root()/'campaign_sources'/'generation'/campaign_id/'uploaded'
    if root.exists(): shutil.rmtree(root)
    (root/'idf').mkdir(parents=True); (root/'epw').mkdir(parents=True)
    buildings=[]; weather=[]
    for info,kind,name in accepted:
        target=root/kind/name; target.write_bytes(z.read(info)); ref=SourceFileRef(source_id=f'{kind}/{name}',name=target.stem,path=str(target.resolve()),sha256=sha256_file(target),building_type=target.stem if kind=='idf' else None)
        (buildings if kind=='idf' else weather).append(ref)
    return tuple(buildings),tuple(weather),root
