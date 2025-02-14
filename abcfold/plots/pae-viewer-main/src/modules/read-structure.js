const codeToNucleotide = {
    A: 'ADE',
    G: 'GUA',
    C: 'CYT',
    T: 'THY',
    U: 'URA',
}

const codeToElement = {
    C: 'Carbon',
    H: 'Hydrogen',
    N: 'Nitrogen',
    O: 'Oxygen',
    P: 'Phosphorus',
    S: 'Sulfur'
}

export function readStructure(file, labels) {
    return (new NGL.Stage()).loadFile(
        file, {name: 'structure'}
    ).catch(error => {
        let message = `Couldn't read file '${file.name}'!`;

        if (file?.name && file.name.slice(-3).toLowerCase() === 'cif') {
            message += " If you're using the PDBx/mmCIF file format, please"
                + " note that the NGL Viewer support for this format is"
                + " unfortunately unstable and might fail.";
        }

        throw {
            type: 'structure-input-error',
            message: message,
            file: file,
            cause: error,
        };
    }).then(component => {
        const chains = [];
        let offset = 0;
        let paeIndex = 0;
        const modifications = [];

        component.structure.eachChain(chain => {
            const sequence = [];
            const residueProxy = chain.structure.getResidueProxy(chain.residueOffset);

            if (residueProxy.isPolymer()) {
                chain.eachResidue(r => {
                    let code;
                    let name;

                    if (r.isProtein()) {
                        code = r.getResname1();
                        name = r.resname;

                        // handle modified residues (PTMs)
                        if (code === 'X') {
                            modifications.push({
                                name: name,
                                index: paeIndex,
                                atomCount: r.atomCount,
                            })
                            paeIndex += r.atomCount - 1;
                        }
                    } else if (r.isDna()) {
                        code = r.resname[1];
                        name = codeToNucleotide[code] ?? code;
                    } else {  // RNA
                        code = r.resname[0];
                        name = codeToNucleotide[code] ?? code;
                    }

                    sequence.push({
                        code: code,
                        name: name
                    });
                    paeIndex++;
                });
            } else {
                chain.eachAtom(a => {
                    const code = a.atomname[0];

                    sequence.push({
                        code: code,
                        name: codeToElement[code] ?? code
                    });
                    paeIndex++;
                });
            }

            let type;

            if (residueProxy.isProtein()) {
                type = 'protein';
            } else if (residueProxy.isDna() || residueProxy.isRna()) {
                type = 'nucleotide';
            } else {
                type = 'molecule';
            }

            chains.push({
                chain: chain.chainname,
                length: sequence.length,
                sequence: sequence,
                offset: offset,
                type: type
            });

            offset += sequence.length;
        });

        if (labels === null) {
            labels = chains.map(chain => chain.chain);
        } else if (labels.length !== chains.length) {
            throw {
                type: 'invalid-chain-labels',
                message: `The number of labels (${labels.length}) doesn't`
                    + ` match the number of chains in the structure`
                    + ` (${chains.length})!`,
                labelsLength: labels.length,
                chainsLength: chains.length
            };
        }

        return {
            chains: chains.map((chain, i) => ({
                ...chain,
                id: labels[i],
                title: labels[i],
                uniprot: labels[i],
                index: i,
                position: i,
            })),
            modifications: modifications
        }
    });
}
