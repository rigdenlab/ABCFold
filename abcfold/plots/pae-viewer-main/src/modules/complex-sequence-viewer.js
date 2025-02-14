import {getChainMapping} from "./structure-meta.js";
import {subunitColors} from "./structure-style.js";

export class ComplexSequenceViewer {
    #root;
    #template;
    #element;

    #container;
    #labels = [];

    #colors = subunitColors;

    #complex = null;
    #subunitsById = null;
    #sequenceElements = [];
    #highlighted = null;
    #selecting = null;
    #selected = null;
    #selectedRanges = null;
    #statusHighlighted = null;
    #statusSelecting = null;
    #statusSelected = null;

    static codeMapping = {
        A: 'Ala', R: 'Arg', N: 'Asn', D: 'Asp', B: 'Asx', C: 'Cys',
        E: 'Glu', Q: 'Gln', Z: 'Glx', G: 'Gly', H: 'His', I: 'Ile',
        L: 'Leu', K: 'Lys', M: 'Met', F: 'Phe', P: 'Pro', S: 'Ser',
        T: 'Thr', W: 'Trp', Y: 'Tyr', V: 'Val'
    };

    constructor(root, template, complex = null) {
        this.#root = root;
        this.#template = template;
        this.#element = template.content.querySelector(
            '.complex-sequence-viewer'
        ).cloneNode(true);
        this.#container = this.#element.querySelector('.csv-container');
        this.#root.appendChild(this.#element);

        if (complex) {
            this.load(complex);
        }
    }

    fetch(resource, options) {
        fetch(resource, options)
            .then(response => response.json())
            .then(this.load.bind(this));
    }

    load(complex) {
        this.#complex = complex;

        // workaround to give unique IDs and names
        const uniprotCounts = new Map();

        for (const member of this.#complex.members) {
            const count = (uniprotCounts.get(member.uniprot) ?? 0) + 1;
            uniprotCounts.set(member.uniprot, count);

            if (count > 1) {
                member.uniprot = `${member.uniprot}#${count}`;
                member.title = `${member.title}#${count}`;
            }
        }

        if (this.#complex.members[0]?.sequence) {
            this.#subunitsById = new Map(
                this.#complex.members.map(member => [member.uniprot, member])
            );

            this.#setupPaeViewerListeners();
            this.#displaySequences();

            return Promise.resolve();
        } else {
            return fetch(
                `crosslinks/${complex.handle}/${complex.structureFile}`
            ).then(response => response.text()).then(fileContent => {
                const sequences = ComplexSequenceViewer.readSequenceFromPDB(
                    fileContent
                );

                this.#complex.chains = getChainMapping(this.#complex);

                let offset = 0;

                for (const [i, member] of this.#complex.members.entries()) {
                    member.chain = this.#complex.chains[member.uniprot];
                    member.sequence = sequences.get(member.chain);
                    member.index = i;

                    member.offset = offset;
                    offset += member.sequence.length;
                }

                this.#subunitsById = new Map(
                    this.#complex.members.map(member => [member.uniprot, member])
                );

                this.#setupPaeViewerListeners();
                this.#displaySequences();
            });
        }
    }

    #setupPaeViewerListeners() {
        const matches = event => (
            event.detail.complex?.id === this.#complex.id
        );

        const getSelection = (subunitId, relativeIndex) =>
            this.#sequenceElements[
            this.#subunitsById.get(subunitId).offset + relativeIndex - 1
                ];

        const createRangeSelection = (selectionStart, selectionEnd) => ({
            start: selectionStart,
            end: selectionEnd
        });

        const selectionFromRanges = (rangeStart, rangeEnd) =>
            createRangeSelection(
                getSelection(...rangeStart),
                getSelection(...rangeEnd)
            );

        document.addEventListener('pv-select-residue-pair', event => {
            if (!matches(event)) return;

            const x = getSelection(...event.detail.selection.x.residue);
            const y = getSelection(...event.detail.selection.y.residue);

            this.#selectRanges(
                createRangeSelection(x, x),
                createRangeSelection(y, y),
            );
        });

        document.addEventListener('pv-select-residue-range', event => {
            if (!matches(event)) return;

            const selection = event.detail.selection;
            const overlap = selection.overlap ?
                selectionFromRanges(...selection.overlap.range) : null;

            this.#selectRanges(
                selectionFromRanges(...selection.x.range),
                selectionFromRanges(...selection.y.range),
                overlap
            );
        });

        document.addEventListener('pv-select-crosslink', event => {
            if (!matches(event)) return;

            const selection = event.detail.selection;

            const residue1 = getSelection(...selection.residue1);
            const residue2 = getSelection(...selection.residue2);

            this.#selectRanges(
                createRangeSelection(residue1, residue1),
                createRangeSelection(residue2, residue2)
            );
        });

        document.addEventListener('pv-reset-selection', event => {
            if (!matches(event)) return;

            this.#resetSelected();
            this.#resetSelectedRanges();
        });
    }

    /**
     *
     * See
     * https://www.cgl.ucsf.edu/chimera/docs/UsersGuide/tutorials/pdbintro.html
     * for details.
     *
     * @param fileContent
     * @returns {Map<any, any>}
     */
    static readSequenceFromPDB(fileContent) {
        const iupacMapping = new Map([
            ["ALA", "A"], ["ASX", "B"], ["CYS", "C"], ["ASP", "D"],
            ["GLU", "E"], ["PHE", "F"], ["GLY", "G"], ["HIS", "H"],
            ["ILE", "I"], ["LYS", "K"], ["LEU", "L"], ["MET", "M"],
            ["ASN", "N"], ["PRO", "P"], ["GLN", "Q"], ["ARG", "R"],
            ["SER", "S"], ["THR", "T"], ["VAL", "V"], ["TRP", "W"],
            ["XAA", "X"], ["TYR", "Y"], ["GLX", "Z"],
        ])

        const sequences = new Map();
        let lastChain = null
        let lastResidueNumber = null;

        for (const line of fileContent.split('\n')) {
            if (!line.startsWith('ATOM')) continue;

            const chain = line.charAt(21);
            const residueNumber = line.slice(22, 26);

            if (residueNumber === lastResidueNumber && chain === lastChain) {
                continue;
            }

            lastChain = chain;
            lastResidueNumber = residueNumber;

            const name = line.slice(17, 20);
            const code = iupacMapping.get(name);
            const residue = {
                code: code,
                name: name
            }

            if (sequences.has(chain)) {
                sequences.get(chain).push(residue);
            } else {
                sequences.set(chain, [residue]);
            }
        }

        return sequences;
    }

    #displaySequences() {
        const colors = this.#colors.map(color => color + '40');

        for (const [m, member] of this.#complex.members.entries()) {
            const subunit = document.createElement('div');
            subunit.classList.add('csv-subunit');

            const label = document.createElement('div');
            label.classList.add('csv-label');

            const typeToUnit = {
                'protein': 'aa',
                'nucleotide': 'nucleotides',
                'molecule': 'atoms',
            }

            const type = member.type ?? 'protein';
            const unit = typeToUnit[type];

            label.textContent
                = `${member.title} (${member.sequence.length} ${unit})`;
            subunit.appendChild(label);
            this.#labels.push(label);

            const sequence = document.createElement('div');
            sequence.classList.add('csv-sequence');
            sequence.style.backgroundColor = colors[m % colors.length];

            for (const [i, residue] of member.sequence.entries()) {
                const segment = document.createElement('div');
                segment.classList.add('csv-segment')

                if (i % 10 === 0) {
                    const index = document.createElement('div');
                    index.classList.add('csv-index')
                    index.textContent = i + 1;
                    segment.appendChild(index);
                }

                const code = document.createElement('div');
                code.classList.add('csv-code')
                code.textContent = residue.code;
                code.dataset.subunit = member.uniprot;
                code.dataset.index = i;
                segment.appendChild(code);

                this.#sequenceElements.push(
                    this.#createResidueSelectionFromElement(code)
                );

                sequence.appendChild(segment);
            }

            subunit.appendChild(sequence);
            this.#container.appendChild(subunit);
        }

        const statusBar = document.createElement('div');
        statusBar.classList.add('csv-status-bar');

        this.#statusSelected = document.createElement('span');
        this.#statusSelected.classList.add('csv-status-text');
        statusBar.appendChild(this.#statusSelected);

        this.#statusSelecting = document.createElement('span');
        this.#statusSelecting.classList.add('csv-status-text');
        statusBar.appendChild(this.#statusSelecting);

        this.#statusHighlighted = document.createElement('span');
        this.#statusHighlighted.classList.add('csv-status-text');
        statusBar.appendChild(this.#statusHighlighted);

        this.#element.appendChild(statusBar);

        this.#container.addEventListener('mouseover', event => {
            const segment = event.target.closest('.csv-segment');

            if (segment === null) {
                this.#resetHighlight();
                return;
            }

            this.#setHighlight(segment.querySelector('.csv-code'));

            if (this.#selecting !== null) {
                this.#setSelecting({
                    start: this.#selecting.start,
                    end: this.#highlighted
                });
            }
        });

        this.#container.addEventListener('mousedown', () => {
            if (this.#highlighted === null) {
                return;
            }

            this.#setSelecting({
                start: this.#highlighted,
                end: this.#highlighted
            });
        });

        this.#container.addEventListener('mouseup', () => {
            if (this.#selecting === null) {
                return;
            }

            this.#setSelected(this.#selecting);
        });
    }

    #residueToString(r) {
        return `${r.residue.name} ${r.index + 1} (${r.subunit.title})`;
    }

    #rangeToString(range) {
        return range.start.element === range.end.element ?
            this.#residueToString(range.start, range.start.subunit.type)
            : `${this.#residueToString(range.start, range.start.subunit.type)}`
            + ` - ${this.#residueToString(range.end, range.end.subunit.type)}`
    }

    * #getResiduesFromSelection(selection) {
        yield* this.#getResiduesFromRange(
            selection.start.absoluteIndex,
            selection.end.absoluteIndex
        )
    }

    * #getResiduesFromRange(start, end) {
        [start, end] = [start, end].sort((a, b) => (a - b));

        for (let i = start; i <= end; i++) {
            yield this.#sequenceElements[i];
        }
    }

    #createResidueSelectionFromElement(element) {
        return this.#createResidueSelection(
            this.#subunitsById.get(element.dataset.subunit),
            parseInt(element.dataset.index),
            element
        );
    }

    #createResidueSelection(subunit, relativeIndex, element) {
        return {
            subunit: subunit,
            index: relativeIndex,
            residue: subunit.sequence[relativeIndex],
            absoluteIndex: subunit.offset + relativeIndex,
            element: element,
        };
    }

    #setHighlight(element) {
        this.#resetHighlight();

        element.classList.add('csv-code-highlighted')
        this.#highlighted = this.#createResidueSelectionFromElement(element);

        this.#setHighlightStatus(
            this.#residueToString(
                this.#highlighted, this.#highlighted.subunit.type
            )
        );
    }

    #resetHighlight() {
        if (this.#highlighted === null) {
            return;
        }

        this.#highlighted.element.classList.remove('csv-code-highlighted');
        this.#highlighted = null;
        this.#resetHighlightStatus();
    }

    #setSelecting(selection) {
        this.#resetSelecting();
        this.#selecting = selection;

        for (const residue of this.#getResiduesFromSelection(selection)) {
            residue.element.classList.add('csv-code-selecting');
        }

        this.#setSelectingStatus(this.#rangeToString(selection));
    }

    #resetSelecting() {
        this.#resetSelected();

        if (this.#selecting === null) {
            return;
        }

        for (const residue of this.#getResiduesFromSelection(this.#selecting)) {
            residue.element.classList.remove('csv-code-selecting');
        }

        this.#selecting = null;
        this.#resetSelectingStatus();
    }

    #setSelected(selection) {
        this.#resetHighlight();
        this.#resetSelecting();
        this.#resetSelected();
        this.#resetSelectedRanges();

        this.#selected = selection;

        for (const residue of this.#getResiduesFromSelection(selection)) {
            residue.element.classList.add('csv-code-selected');
        }

        this.#setSelectionStatus(this.#rangeToString(selection));

        const range = [selection.start, selection.end].sort(
            (a, b) => a.absoluteIndex - b.absoluteIndex
        ).map(residue => [
            residue.subunit.uniprot, residue.index + 1
        ]);

        this.#element.dispatchEvent(new CustomEvent('csv-select-range', {
            bubbles: true,
            detail: {
                complex: this.#complex,
                selection: {range: range, color: 'magenta'}
            }
        }));
    }

    #resetSelected() {
        if (this.#selected === null) {
            return;
        }

        for (const residue of this.#getResiduesFromSelection(this.#selected)) {
            residue.element.classList.remove('csv-code-selected');
        }

        this.#selected = null;
        this.#resetSelectionStatus();
    }

    #selectRanges(x, y, overlap = null) {
        this.#resetHighlight();
        this.#resetSelecting();
        this.#resetSelected();
        this.#resetSelectedRanges();

        this.#selectedRanges = {x: x, y: y, overlap: overlap};

        const selections = [
            [x, 'csv-code-selected-x'],
            [y, 'csv-code-selected-y'],
        ];

        if (overlap !== null) {
            selections.push([overlap, 'csv-code-selected-overlap', 'overlap']);
        }

        for (const [selection, cssClass] of selections) {
            for (const residue of this.#getResiduesFromSelection(selection)) {
                residue.element.classList.add(cssClass);
            }
        }

        const minIndex = Math.min(...selections.map(
            ([selection, _]) => Math.min(
                selection.start.absoluteIndex,
                selection.end.absoluteIndex
            )
        ));

        const first = this.#sequenceElements[minIndex].element.parentElement;
        this.#container.scrollTo({
            top: first.offsetTop - this.#container.offsetTop
                - this.#labels[0].offsetHeight,
            behavior: 'smooth'
        });

        const span = (text, cssClass) =>
            `<span class="${cssClass}">${text}</span>`;

        let message = span(`X: ${this.#rangeToString(x)}`, 'csv-x')
            + ', ' + span(`Y: ${this.#rangeToString(y)}`, 'csv-y');

        if (overlap !== null) {
            message += ', ' + span(
                `Overlap: ${this.#rangeToString(overlap)}`, 'csv-overlap'
            );
        }

        this.#setSelectionStatus(message);
    }

    #resetSelectedRanges() {
        if (this.#selectedRanges === null) {
            return;
        }

        const ranges = [
            [this.#selectedRanges.x, 'csv-code-selected-x'],
            [this.#selectedRanges.y, 'csv-code-selected-y'],
        ];

        if (this.#selectedRanges.overlap !== null) {
            ranges.push([
                this.#selectedRanges.overlap, 'csv-code-selected-overlap'
            ]);
        }

        for (const [selection, cssClass] of ranges) {
            for (const residue of this.#getResiduesFromSelection(selection)) {
                residue.element.classList.remove(cssClass);
            }
        }

        this.#selectedRanges = null;
        this.#resetSelectionStatus();
    }

    #setHighlightStatus(message) {
        this.#statusHighlighted.innerHTML = '<b>Highlighted</b>: ' + message;
    }

    #resetHighlightStatus() {
        this.#statusHighlighted.textContent = "";
    }

    #setSelectingStatus(message) {
        this.#statusSelecting.innerHTML = '<b>Selecting</b>: ' + message;
    }

    #resetSelectingStatus() {
        this.#statusSelecting.textContent = "";
    }

    #setSelectionStatus(message) {
        this.#statusSelected.innerHTML = '<b>Selected</b>: ' + message;
    }

    #resetSelectionStatus() {
        this.#statusSelected.textContent = "";
    }
}
