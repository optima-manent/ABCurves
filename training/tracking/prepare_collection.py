"""Prepare the complete selected tracking cohorts from raw release collections."""
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace

from abcurves.capture_archives import archive_sources, sha256
from .support import read, write
from .tracking_prepare import prepare, assemble_original, assemble_new


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='Collection ZIP or folder containing all tracking parts/session ZIPs')
    parser.add_argument('output', type=Path)
    parser.add_argument('--validator', type=Path, required=True)
    args = parser.parse_args()
    contract_path = Path(__file__).with_name('frozen_sources.json')
    person3_path = Path(__file__).with_name('person3_source.json')
    contract, person3 = read(contract_path), read(person3_path)
    wanted = {row['source_zip_sha256']: (person, contract_path)
              for person, row in contract['old_calibrations'].items()}
    wanted.update({digest: (name, contract_path) for name, digest in contract['new_archive_sha256'].items()})
    wanted[person3['archive_sha256']] = ('person3', person3_path)
    args.output.mkdir(parents=True, exist_ok=False)
    found, excluded = {}, []
    write(args.output/'collection.json', {'status':'BUILDING'})
    with archive_sources(args.input) as sources:
        for archive, lineage in sources:
            if archive.is_dir():
                raise ValueError('Selected tracking reproduction requires original session ZIPs; preserve them when extracting release parts')
            digest = sha256(archive)
            if digest not in wanted:
                excluded.append({'lineage':lineage,'archive_sha256':digest,'reason':'outside_selected_six_source_training_cohort'})
                continue
            name, source_contract = wanted[digest]
            if name in found: raise ValueError('Duplicate selected archive: '+name)
            output = args.output/name
            prepare(SimpleNamespace(archive=archive, source=name, output=output,
                    validator=args.validator.resolve(), reuse_projection=None, contract=source_contract))
            found[name] = {'lineage':lineage, 'archive_sha256':digest}
    missing = sorted({name for name,_ in wanted.values()}-found.keys())
    if missing:
        write(args.output/'collection.json',{'status':'INCOMPLETE','sources':found,'excluded':excluded,'missing':missing})
        raise ValueError('Missing selected raw sources: '+', '.join(missing))
    assemble_original(SimpleNamespace(inputs=[args.output/p for p in ('person1','person2','person3')],
                                      output=args.output/'original'))
    assemble_new(SimpleNamespace(inputs=[args.output/p for p in contract['new_archive_sha256']],
                                 output=args.output/'new', contract=contract_path))
    write(args.output/'collection.json',{'status':'COMPLETE','sources':found,'excluded':excluded,
        'contract_sha256':sha256(contract_path),'person3_contract_sha256':sha256(person3_path)})


if __name__ == '__main__': main()
