import {partitionOn, range} from "./utils.js";
import {
    crosslinkStyleCommon,
    crosslinkStyleDeselected,
    crosslinkStyleSelected,
    defaultStyles
} from "./structure-style.js";

const State = Object.freeze({
    LOADING: Symbol('loading'),
    DONE: Symbol('done'),
    ERROR: Symbol('error')
});

export class StructureViewer {
    #root;
    #template;
    #element;
    #display;
    #loadingOverlay;
    #errorOverlay;
    #errorMessage;
    #captionBox;
    #caption;
    #screenshot;

    #meta;
    #showCaption;
    #stage;
    #structureComponent;
    #styles = new Map();
    #currentStyle = null;
    #crosslinks = {all: [], schemes: []};
    #state;
    #currentHighlightedDistance = null;

    constructor(
        root, template, meta, size = null, style = null, showCaption = true
    ) {
        this.#root = root;
        this.#template = template;
        this.#element = this.#template.content.querySelector(
            '.structure-viewer'
        ).cloneNode(true);

        this.#loadingOverlay
            = this.#element.querySelector('.sv-loading');
        this.#errorOverlay
            = this.#element.querySelector('.sv-error');

        this.#display = this.#element.querySelector('.sv-display');

        this.#errorMessage = this.#errorOverlay.querySelector(
            '.sv-error-message'
        );

        this.#captionBox = this.#element.querySelector('.sv-caption');
        this.#caption = this.#captionBox.querySelector('a');

        this.#screenshot = this.#element.querySelector('.sv-screenshot');

        this.#root.appendChild(this.#element);

        this.#showCaption = showCaption;
        this.#stage = new NGL.Stage(this.#display);
        this.#stage.setParameters({backgroundColor: 'white'});

        if (size !== null) {
            this.setSize(...size);
        }

        this.setStructure(meta, style);

        this.#screenshot.addEventListener('click', this.saveImage.bind(this));
    }

    saveImage() {
        if (!this.#meta.complex) {
            return;
        }

        saveAs(
            this.#element.querySelector('canvas').toDataURL('image/png'),
            this.#getDownloadName() + '.png'
        );

        // this.#stage.makeImage({
        //     trim: false,
        //     factor: 1,
        //     antialias: true,
        //     transparent: false
        // }).then(image => {
        //     NGL.download(image, this.#getDownloadName() + '.png');
        // });
    }

    #getDownloadName() {
        if (this.#meta.complex?.structureFile) {
            const name = this.#meta.complex.structureFile.name
                ?? this.#meta.complex.structureFile;
            return name.split('/').pop()
                .slice(0, name.lastIndexOf('.'));
        } else {
            return this.#meta.complex.members.map(
                member => member.title
            ).join('-');
        }
    }

    #setupSelectionControls() {
        this.#render(defaultStyles.get('selection'), false);
    }

    #setupSelectionListeners() {
        const matches = event => (
            event.detail.complex?.id === this.#meta.complex.id
        );

        document.addEventListener('pv-select-residue-pair', event => {
            if (!matches(event)) return;

            this.selectResiduePair(
                event.detail.selection.x, event.detail.selection.y
            );
        });

        document.addEventListener('pv-select-residue-range', event => {
            if (!matches(event)) return;

            this.selectResidueRanges([
                event.detail.selection.x,
                event.detail.selection.y,
                event.detail.selection.overlap
            ]);
        });

        document.addEventListener('pv-select-crosslink', event => {
            if (!matches(event)) return;

            this.selectCrosslink(event.detail.selection.id);
        });

        document.addEventListener('pv-select-region', event => {
            if (!matches(event)) return;

            this.selectResidueRanges(event.detail.type === 'single' ?
                [event.detail.selection]
                : [
                    event.detail.selection.x,
                    event.detail.selection.y
                ]
            );
        });

        document.addEventListener('pv-reset-selection', event => {
            if (!matches(event)) return;

            this.resetSelection();
        });

        document.addEventListener('csv-select-range', event => {
            if (!matches(event)) return;

            this.selectResidueRanges([event.detail.selection]);
        });
    }

    setStructure(meta, style = null) {
        this.#stage.removeAllComponents();


        this.#meta = meta;

        if (this.#showCaption) {
            this.#captionBox.classList.add('hidden');
            this.#updateCaption();
        }


        return this.#load(
            this.#meta.url
        ).then(() => {
            if (style !== null) {
                this.setStyle(style).then(() => {
                    if (this.#meta.complex) {
                        this.#setupSelectionControls();
                        this.#setupSelectionListeners();
                    }
                });
            }

            if (this.#meta.crosslinks.length > 0) {
                this.#renderCrosslinks(style);
                this.setCrosslinkScheme('restraint');
            }
        });
    }

    #updateCaption() {
        const caption = new Map([
            ['pdb', [
                `http://www.rcsb.org/structure/${this.#meta.id}`,
                `${this.#meta.id} (PDB)`
            ]],
            ['alphafold', [
                `https://alphafold.ebi.ac.uk/entry/${this.#meta.id}`,
                `${this.#meta.id} (AlphaFold)`
            ]],
            ['predicted-complex', [
                `predictedComplex?id=${this.#meta.id}`,
                this.#meta.complex?.handle
            ]],
        ]).get(this.#meta.source);

        if (caption) {
            this.#captionBox.classList.remove('hidden');
            this.#caption.href = caption[0];
            this.#caption.textContent = caption[1];
        } else {
            throw (
                `Structure source '${this.#meta.source}'`
                + ` of '${this.#meta.id}' invalid!`
            );
        }
    }

    #load(url) {
        this.#setState(State.LOADING);

        return this.#stage.loadFile(
            url, {name: 'structure'}
        ).then(structure => {
            this.#structureComponent = structure;
            structure.autoView();
            this.#stage.viewerControls.zoom(-0.1);
            this.#setState(State.DONE);
        }).catch(error => this.#showError(
            `Couldn't load structure '${this.#meta.id}'!`, [error, url]
        ));
    }

    #renderCrosslinks() {
        const [intraLinks, interLinks] = partitionOn(
            this.#meta.crosslinks,
            link => link.get('Protein1') === link.get('Protein2')
        );

        const [satisfiedLinks, violatedLinks] = partitionOn(
            this.#meta.crosslinks,
            link =>
                !link.has('RestraintSatisfied')
                || link.get('RestraintSatisfied').toLowerCase() === 'true'
        );

        const crosslinkReprs = this.#meta.crosslinks.map(crosslink =>
            this.#structureComponent.addRepresentation('distance', {
                ...crosslinkStyleCommon,
                name: `crosslink-${crosslink.get('id')}`,
                atomPair: [crosslink.get('atoms')]
            })
        );

        const filterReprs = (selection, color) => ({
            links: this.#meta.crosslinks.filter(crosslink =>
                selection.map(s => s.get('id')).includes(crosslink.get('id'))
            ).map(crosslink => crosslinkReprs[crosslink.get('id')]),
            color: color
        });

        this.#crosslinks = {
            all: crosslinkReprs,
            schemes: {
                restraint: new Map([
                    ['satisfied', filterReprs(satisfiedLinks, 'blue')],
                    ['violated', filterReprs(violatedLinks, 'red')]
                ]),
                connection: new Map([
                    ['intra', filterReprs(intraLinks, 'grey')],
                    ['inter', filterReprs(interLinks, 'orange')]
                ])
            }
        }
    }

    setCrosslinkScheme(scheme) {
        if (!scheme) {
            for (const repr of this.#crosslinks.all) {
                repr.setVisibility(false);
            }
        } else {
            for (const group of this.#crosslinks.schemes[scheme].values()) {
                const params = {
                    color: group.color,
                    labelBackgroundColor: group.color
                };

                for (const repr of group.links) {
                    repr.setParameters(params);
                    repr.setVisibility(true);
                }
            }
        }
    }

    /**
     * Sets style for representation.
     *
     * While loading, ignores request and returns rejecting promise.
     *
     * @param style
     * @returns {Promise<unknown>}
     */
    setStyle(style) {
        return new Promise((resolve, reject) => {
            if (this.#state !== State.DONE) {
                reject();
            }

            if (this.#styles.get(style.name)) {
                this.#showStructureRepresentation(style.name);
                resolve();
            } else {
                return this.#render(style, true)
                    .then(() => this.#showStructureRepresentation(style.name))
                    .then(() => resolve())
            }
        })

    }

    #showStructureRepresentation(name) {
        if (this.#currentStyle !== null) {
            for (const repr of this.#styles.get(this.#currentStyle).reprs) {
                repr.setVisibility(false);
            }

        }

        this.#currentStyle = name;

        for (const repr of this.#styles.get(this.#currentStyle).reprs) {
            repr.setVisibility(true);
        }
    }

    #render(style, visible = false) {
        this.#setState(State.LOADING);
        return style.render(
            this.#structureComponent, this.#stage, this.#meta, true, visible
        ).then(representations => {
            this.#styles.set(style.name, {
                style: style, reprs: representations
            });
            this.#setState(State.DONE);
        }).catch(error => this.#showError(
            `Couldn't render representation '${style.name}'`
            + ` for structure '${this.#meta.id}'`,
            [style, error]
        ));
    }

    #setState(state) {
        this.#state = state;

        for (const [s, element] of new Map([
            [State.LOADING, this.#loadingOverlay],
            [State.ERROR, this.#errorOverlay],
        ])) {
            element.classList.toggle('hidden', s !== state);
        }
    }

    #showError(message, details = null) {
        this.#errorMessage.textContent = message;

        console.error(message);

        if (details !== null) {
            console.error(details);
        }

        this.#setState(State.ERROR);
    }

    /**
     * Sets size of the NGL stage canvas. `Stage.resize()` is not
     * used, as it relies on the bounding rect of the parent element,
     * which can lead to issues as it may be hidden sometimes.
     *
     * @param width
     * @param height
     */
    setSize(width, height) {
        this.#stage.viewer.setSize(width, height);
    }

    getMeta() {
        return this.#meta;
    }

    resetSelection() {
        for (const repr of this.#styles.get(this.#currentStyle).reprs) {
            repr.setVisibility(true);
        }

        for (const repr of this.#styles.get('selection').reprs) {
            repr.setVisibility(false);
        }

        for (const repr of this.#crosslinks.all) {
            repr.setParameters(crosslinkStyleSelected);
        }

        if (this.#currentHighlightedDistance !== null) {
            this.#structureComponent.removeRepresentation(
                this.#currentHighlightedDistance
            );
        }

        this.#currentHighlightedDistance = null;
    }

    #highlightSelectedResidues(selectionScheme) {
        for (const repr of this.#styles.get(this.#currentStyle).reprs) {
            repr.setVisibility(false);
        }

        this.#fadeOutCrosslinks();

        const scheme
            = NGL.ColormakerRegistry.addSelectionScheme(selectionScheme);

        for (const repr of this.#styles.get('selection').reprs) {
            repr.setVisibility(true);
            repr.setColor(scheme);
        }
    }

    #fadeOutCrosslinks() {
        for (const repr of this.#crosslinks.all) {
            repr.setParameters(crosslinkStyleDeselected);
        }
    }

    /**
     * Returns member of complex by ID.
     *
     * As a workaround for the example page, also use the UniProt ID
     * for matching.
     */
    #getMemberById(id) {
        return this.#meta.complex.members.find(m => m.id === id || m.uniprot === id);
    }

    /**
     * Returns atom index for molecule by chain label and index
     * (undefined behavior for chains which correspond not to molecules
     * but to polymers like proteins, DNA, RNA etc.).
     *
     * @param chainLabel
     * @param index
     * @returns {number}
     */
    #getAtomIndexForMolecule(chainLabel, index) {
        let offset = null;

        this.#structureComponent.structure.eachChain(c => {
            if (c.chainname === chainLabel) {
                offset = c.atomOffset;
            }
        });

        return offset + index - 1;
    }

    /**
     *  Returns NGL selection string for a chain and a coordinate (or a
     *  pair of coordinates) within that chain. If chain corresponds to
     *  a molecule, the coordinates are atom indices (e.g. `@1,2,3`),
     *  otherwise residues indices (e.g. `1-5:A`).
     */
    #getSele(member, start, end = undefined) {
        // workaround for examples
        const chainLabel = member.chain ?? this.#meta.complex.chains[member.uniprot];

        if (member.type === 'molecule') {
            const startIndex = this.#getAtomIndexForMolecule(chainLabel, start);
            let sele = `@${startIndex}`;

            if (end !== undefined) {
                const endIndex = this.#getAtomIndexForMolecule(chainLabel, end);

                if (endIndex > startIndex) {
                    sele += ',' + [...range(startIndex, endIndex + 1)].join(',')
                }
            }

            return sele;
        } else {
            let residues = start;

            if (end !== undefined) {
                residues += '-' + end;
            }

            return `${residues}:${chainLabel}`;
        }
    }

    selectResiduePair(selectionX, selectionY) {
        this.resetSelection();

        const pair = [selectionX.residue, selectionY.residue].map(res => {
            const [id, index] = res;
            return this.#getSele(this.#getMemberById(id), index);
        });

        const [seleX, seleY] = pair;

        this.#highlightSelectedResidues([
            [selectionX.color, seleX],
            [selectionY.color, seleY],
            ['lightgrey', '*']
        ]);

        this.#currentHighlightedDistance
            = this.#structureComponent.addRepresentation('distance', {
            labelSize: 2,
            labelBackground: true,
            labelBackgroundColor: 'seagreen',
            labelColor: 'white',
            linewidth: 8,
            color: 'seagreen',
            labelUnit: 'angstrom',
            atomPair: [pair],
            name: 'highlighted-distance-' + pair.join('-')
        });

        this.#structureComponent.autoView(pair.join(' or '), 500);
    }

    selectResidueRanges(selections) {
        this.resetSelection();

        selections = selections.filter(selection => selection !== null);

        const chains = this.#meta.complex.chains;
        const members = this.#meta.complex.members;

        const getRangeSelection = range => {
            if (range === null) {
                return [];
            }

            const [[uniprot1, residue1], [uniprot2, residue2]] = range;

            const [memberIndex1, memberIndex2] = [uniprot1, uniprot2].map(
                uniprot => members.findIndex(m => m.uniprot === uniprot)
            );

            const member1 = members[memberIndex1];
            const member2 = members[memberIndex2];

            if (uniprot1 === uniprot2) {
                return this.#getSele(member1, residue1, residue2);
            }

            // if residues not in same subunit (denoted by UniProt id)
            // include all residues of other subunits included in the
            // range
            const selections = [
                this.#getSele(member1, residue1, member1.length),
                this.#getSele(member2, 1, residue2)
            ];

            for (const member of members.slice(memberIndex1 + 1, memberIndex2)) {
                selections.push(`:${chains[member.uniprot]}`);
            }

            return selections.join(' or ');
        };

        for (const selection of selections) {
            selection.sele = getRangeSelection(selection.range);
        }

        this.#highlightSelectedResidues([
            ...selections.map(selection => [selection.color, selection.sele]),
            ['lightgrey', '*']
        ]);

        const sele = selections.map(
            selection => selection.sele
        ).join(' or ');

        this.#structureComponent.autoView(sele, 500);
    }

    selectCrosslink(id) {
        if (!(id in this.#crosslinks.all)) {
            throw `Can't find crosslink '${id}'!`;
        }

        this.resetSelection();
        this.#fadeOutCrosslinks();

        const repr = this.#crosslinks.all[id];

        repr.setParameters(crosslinkStyleSelected);

        const selection = repr.parameters.atomPair[0].join(' or ');

        this.#structureComponent.autoView(selection, 500);
    }
}
