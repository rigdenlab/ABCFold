import * as Utils from './utils.js';
import {subunitColors} from "./structure-style.js";
import {readStructure} from "./read-structure.js";

export class PaeViewer {
    #root;
    #template;
    #element;

    #graph;
    #graphDefs;
    #stripePattern;
    #graphArea;
    #paeMatrix;
    #axesGroup;
    #unitTicksGroup;
    #sequenceTicksGroup;
    #sequenceTickLabelsGroup;
    #unitTickLabelsGroup;
    #dividerGroup;
    #interactiveGroup;
    #selectionGroup;
    #crosslinkGroup;
    #regionGroup;
    #paeScale;
    #paeScaleTicks;
    #statusCursor;
    #statusSelection;
    #regionToggleCheckBox;
    #downloadMatrix;
    #downloadSvg;

    #viewBox;
    #colorMapping = null;
    #colorRange = subunitColors;
    #paePalette = chroma.cubehelix().start(120).rotations(0)
        .hue(0.8).gamma(1).lightness([0.2, 0.95]);

    #dim;
    #crosslinks;
    #crosslinkMarkers;
    #complex;
    #selection;
    #selectedRegion = null;
    #isCursorOnGraph = false;
    #isShiftKeyDown = false;

    #style = {
        general: {
            fontFamily: 'Arial, Helvetica, sans-serif',
        },
        defaults: {
            chartColor: 'black',
            chartLineThickness: '0.2%',
            fontSize: 0.04,
            selectionOutlineColor: 'white',
            markerOutlineColor: 'white',
            markerOutlineThickness: '0.2%',
            markerSize: '1%'
        },
        elements: {
            axes: {
                color: '$chartColor',
                thickness: '$chartLineThickness'
            },
            boxes: {
                roundness: '1%',
                color: 'white',
                opacity: 0.8
            },
            dividers: {
                color: '$chartColor',
                thickness: '0.4%'
            },
            ticks: {
                unitInterval: 100,  // in sequence coordinates
                fontSize: '$fontSize',
                color: '$chartColor',
                thickness: '$chartLineThickness',
                labelGap: 0.01,
                units: {
                    length: 0.02
                },
                subunits: {
                    length: 0.08
                },
            },
            subunitLabels: {
                gap: 0.16,
                fontWeight: 'bold',
                fontStyle: 'italic',
                color: 'white'
            },
            regions: {
                opacity: 0.7,
                fontSize: '$fontSize'
            },
            selection: {
                lines: {
                    color: '$selectionOutlineColor',
                    thickness: '0.5%',
                    dashLength: '0.5%',
                },
                markers: {
                    outlineColor: '$markerOutlineColor',
                    outlineThickness: '$markerOutlineThickness',
                    size: '$markerSize'
                },
                rect: {
                    color: '$selectionOutlineColor',
                    opacity: '0.5'
                },
                colors: {
                    x: 'cyan',
                    y: 'orange',
                    overlap: 'magenta'
                }
            },
            crosslinks: {
                outlineColor: '$markerOutlineColor',
                outlineThickness: '$markerOutlineThickness',
                opacity: 0.75,
                size: '$markerSize',
                restraintColors: {
                    satisfied: 'blue',
                    violated: 'red'
                }
            }
        }
    };

    static codeMapping = {
        A: 'Ala', R: 'Arg', N: 'Asn', D: 'Asp', B: 'Asx', C: 'Cys',
        E: 'Glu', Q: 'Gln', Z: 'Glx', G: 'Gly', H: 'His', I: 'Ile',
        L: 'Leu', K: 'Lys', M: 'Met', F: 'Phe', P: 'Pro', S: 'Ser',
        T: 'Thr', W: 'Trp', Y: 'Tyr', V: 'Val'
    };

    constructor(root, template, style = {}, colorRange = null, paePalette = null) {
        this.#root = root;
        this.#template = template;
        this.#element
            = template.content.querySelector('.pae-viewer').cloneNode(true);

        this.#graph = this.#element.querySelector('.pv-graph');
        this.#graphDefs = this.#graph.querySelector('defs');
        this.#stripePattern = this.#graphDefs.querySelector(
            '#stripes-template'
        );
        this.#graphArea = this.#graph.querySelector('.pv-graph-area');
        this.#paeMatrix = this.#graphArea.querySelector('.pv-pae-matrix');
        this.#axesGroup = this.#graphArea.querySelector('.pv-axes');
        this.#unitTicksGroup = this.#axesGroup.querySelector('.pv-unit-ticks');
        this.#sequenceTicksGroup
            = this.#axesGroup.querySelector('.pv-sequence-ticks');
        this.#sequenceTickLabelsGroup
            = this.#axesGroup.querySelector('.pv-sequence-tick-labels');
        this.#unitTickLabelsGroup
            = this.#axesGroup.querySelector('.pv-unit-tick-labels');
        this.#dividerGroup = this.#graphArea.querySelector('.pv-dividers');
        this.#interactiveGroup = this.#graphArea.querySelector(
            '.pv-interactive-layer'
        );
        this.#selectionGroup = this.#graphArea.querySelector('.pv-selections');
        this.#regionGroup = this.#graphArea.querySelector('.pv-regions');
        this.#crosslinkGroup = this.#graphArea.querySelector('.pv-crosslinks');

        this.#paeScale = this.#element.querySelector('.pv-color-scale');
        this.#paeScaleTicks = this.#element.querySelector('.pv-color-ticks');

        this.#statusCursor = this.#element.querySelector('.pv-status-cursor');
        this.#statusSelection = this.#element.querySelector(
            '.pv-status-selection'
        );

        this.#regionToggleCheckBox = this.#element.querySelector(
            '.pv-region-toggle input'
        );

        this.#downloadMatrix = this.#element.querySelector(
            '.pv-download-matrix'
        );

        this.#downloadSvg = this.#element.querySelector('.pv-download-svg');

        this.#root.appendChild(this.#element);

        const rect = this.#element.getBoundingClientRect();
        this.#viewBox = {width: rect.width, height: rect.height};
        this.#graph.setAttribute('viewBox', `0 0 ${rect.width} ${rect.height}`);
        this.#graph.setAttribute('width', rect.width);
        this.#graph.setAttribute('height', rect.height);

        // workaround for clients not supporting 'transform-origin', e.g. Safari
        const yLabel = this.#graph.querySelector('.pv-axis-y-label');
        yLabel.removeAttribute('transform-origin');
        yLabel.setAttribute(
            'transform',
            `rotate(-90 ${-0.35 * this.#viewBox.width}`
            + ` ${0.5 * this.#viewBox.height})`
        );

        if (colorRange !== null) {
            this.#colorRange = colorRange;
        }

        if (paePalette !== null) {
            this.#paePalette = paePalette;
        }

        this.#updateStyle(style);
        this.#insertStyleDefaults();

        for (const line of this.#axesGroup.querySelectorAll('line')) {
            Utils.setAttributes(line, {
                stroke: this.#style.elements.axes.color,
                'stroke-width': this.#style.elements.axes.thickness
            })
        }

        // in residue coordinates

        this.#dim = 0;
        this.#complex = null;

        this.#resetSelection();

        this.#crosslinkMarkers = [];
    }

    #drawPaeColorScale(maxPae) {
        const colors = [];
        const numSteps = 10;

        for (let i = 0; i <= 10; i++) {
            colors.push(this.#paePalette(i / numSteps).hex());
        }

        const gradient = `linear-gradient(to right, ${colors.join(', ')})`;
        const paeScale = document.querySelector('.pv-color-scale');
        paeScale.style.background = gradient;

        const maxTick = document.createElement('span');
        maxTick.textContent = maxPae;
        maxTick.style.left = '100%';
        this.#paeScaleTicks.appendChild(maxTick);

        const interval = 0.05 * 10 ** Math.ceil(Math.log10(maxPae));

        for (
            let tickValue = 1;
            tickValue * interval <= maxPae - interval;
            tickValue++
        ) {
            const tick = document.createElement('span');
            const value = tickValue * interval;
            tick.textContent = value.toString();
            tick.style.left = `${value / maxPae * 100}%`;
            this.#paeScaleTicks.appendChild(tick);
        }
    }

    #isPrimitive = x => {
        return typeof x === 'string'
            || typeof x === 'boolean'
            || typeof x === 'number';
    };

    #updateStyle(style) {
        const updateObject = (base, update, keys = []) => {
            const [baseKeys, updateKeys] = [base, update].map(obj => new Set(
                Object.entries(obj).map(([key, _]) => key)
            ));

            const unrecognizedKeys = [...new Set([...updateKeys].filter(
                key => !baseKeys.has(key))
            )];

            if (unrecognizedKeys.length > 0) {
                throw `Unrecognized nested style attribute(s)`
                + ` '${unrecognizedKeys.join(', ')}'`
                + (keys.length > 0 ? ` in '${keys.join('.')}'` : '')
                + `!`;
            }

            const sharedKeys = new Set([...baseKeys].filter(
                key => updateKeys.has(key))
            );

            for (const key of sharedKeys) {
                const newKeys = [...keys, key];

                if (typeof base[key] !== typeof update[key]) {
                    throw `Style attribute '${newKeys.join('.')}' with value`
                    + ` '${update[key]}' should be of type`
                    + ` '${typeof base[key]}', but has type`
                    + ` '${typeof update[key]}'!`;
                }

                if (this.#isPrimitive(base[key])) {
                    base[key] = update[key];
                } else {
                    updateObject(base[key], update[key], newKeys);
                }
            }
        };

        updateObject(this.#style, style);
    }

    #insertStyleDefaults() {
        const replacePlaceholders = (style, keys = []) => {
            for (const [key, value] of Object.entries(style)) {
                const newKeys = [...keys, key];

                if (this.#isPrimitive(value)) {
                    if (!(typeof value === 'string' && value.startsWith('$'))) {
                        continue;
                    }

                    const defaultsKey = value.slice(1);

                    if (!this.#style.defaults.hasOwnProperty(defaultsKey)) {
                        throw `Default value '${defaultsKey}' used for style`
                        + ` attribute '${newKeys.join('.')}' doesn't`
                        + ` exist!`;
                    }

                    style[key] = this.#style.defaults[defaultsKey];
                } else {
                    replacePlaceholders(value, newKeys);
                }
            }
        };

        replacePlaceholders(this.#style.elements);
    }

    #resetSelection() {
        this.#selection = {
            rect: null,
            rectMarkers: [],
            startCoords: [],
            startMarkers: [],
            lines: [],
            rangeLines: [],
            rangeMarkers: [],
        }
    }

    getElement() {
        return this.#element;
    }

    fetch(resource, options) {
        fetch(resource, options)
            .then(response => response.json())
            .then(this.load.bind(this));
    }

    #relative(coord) {
        return coord / this.#dim;
    }

    load(complex, colorMapping = null) {
        const directory = `crosslinks/${complex.handle}/`;

        if (Object.hasOwn(complex, 'pae') && Object.hasOwn(complex, 'maxPae')) {
            this.createPaeImage(complex.pae, complex.maxPae).then(url =>
                this.#setPaeImage(url)
            );
        } else if (Object.hasOwn(complex, 'paeImageUrl')) {
            this.#setPaeImage(complex.paeImageUrl);
        } else {
            this.#setPaeImage(directory + complex.handle + '.png');
        }

        this.#drawPaeColorScale(complex.maxPae ?? 31.75);

        if (complex.members && !complex.members[0].sequence) {
            readStructure(directory + complex.structureFile, null).then(structure => {
                for (let i = 0; i < structure.chains.length; i++) {
                    complex.members[i].sequence = structure.chains[i].sequence;
                }
            });
        }

        const sequenceLengths = complex.members.map(member => member.length);
        this.#dim = Utils.sum(sequenceLengths);
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

        this.#colorMapping = colorMapping;
        this.#initColors(complex.members, colorMapping, this.#colorRange);

        this.#addDividers(sequenceLengths);
        this.#addRegions(complex.members);
        this.#addTicks(complex.members);

        if (complex.crosslinks) {
            this.addCrosslinks(complex.crosslinks, complex.members);
        } else if (complex.crosslinksUrl?.name.length > 0) {
            const reader = new FileReader();

            reader.addEventListener('load', () => {
                try {
                    this.addCrosslinks(
                        Utils.readDSV(reader.result, null, ','),
                        complex.members
                    );
                } catch (error) {
                    console.error(error);
                    alert("Couldn't load crosslinks from file!");
                }
            }, false);
            reader.addEventListener('error', () => {
                console.error(error);
                alert("Couldn't load crosslinks from file!");
            }, false);

            reader.readAsText(complex.crosslinksUrl);
        } else if (complex.crosslinksFile) {
            fetch(directory + complex.crosslinksFile)
                .then(response => response.text())
                .then(crosslinkTable => {
                    this.addCrosslinks(
                        Utils.readDSV(crosslinkTable, null, ','),
                        complex.members
                    );
                });
        }

        this.setupSelectionListeners();
        this.#setupSvgDownloadListener();
    }

    #getDownloadName() {
        if (this.#complex?.structureFile) {
            const name = this.#complex.structureFile.name
                ?? this.#complex.structureFile;
            return name.split('/').pop()
                .slice(0, name.lastIndexOf('.'));
        } else {
            return this.#complex.members.map(member => member.title).join('-');
        }
    }

    #setupSvgDownloadListener() {
        this.#downloadSvg.addEventListener('click', _ => {
            const svg = this.#graph.cloneNode(true);

            svg.querySelector('.pv-selections').remove();
            svg.querySelector('.pv-regions').remove();

            // workaround for clients not supporting 'dominant-baseline',
            // e.g. Office applications (Word, PowerPoint)
            for (const text of svg.querySelectorAll(
                'text[dominant-baseline=central]'
            )) {
                const relativeY = parseFloat(text.getAttribute('y'))
                    / 100;
                text.removeAttribute('dominant-baseline');
                text.setAttribute('y', relativeY * this.#viewBox.height + 5);
            }

            const pad = {left: 25, right: 25, top: 25, bottom: 25};
            const bbox = this.#graph.getBBox();
            const width = bbox.width + pad.left + pad.right;
            const height = bbox.height + pad.top + pad.bottom;

            const outerSvg = svg.cloneNode(false);
            outerSvg.setAttribute('viewBox', `0 0 ${width} ${height}`);
            outerSvg.setAttribute('width', width);
            outerSvg.setAttribute('height', height);

            const container = Utils.createSVG('g');
            container.setAttribute(
                'transform',
                `translate(${-bbox.x + pad.left}, ${-bbox.y + pad.top})`
            );

            container.appendChild(svg);
            outerSvg.appendChild(container);

            const imageUrl = this.#downloadMatrix.getAttribute('href');

            fetch(imageUrl).then(response => response.blob()).then(image => {
                const reader = new FileReader();
                reader.readAsDataURL(image);

                reader.onloadend = () => {
                    const paeMatrix = svg.querySelector('.pv-pae-matrix');
                    paeMatrix.setAttribute('xlink:href', reader.result);
                    paeMatrix.removeAttribute('href');

                    saveAs(
                        new Blob(
                            [outerSvg.outerHTML],
                            {type: 'image/svg+xml;charset=utf-8'}
                        ),
                        this.#getDownloadName() + '.svg'
                    );
                }
            });
        });
    }

    #setPaeImage(url) {
        this.#paeMatrix.setAttribute('href', url);
        this.#downloadMatrix.classList.remove('hidden');
        this.#downloadMatrix.setAttribute('href', url);

        this.#downloadMatrix.setAttribute('download', this.#getImageName(url));
    }

    #getImageName(url) {
        if (url.startsWith('blob:')) {
            return this.#getDownloadName() + '_pae.png'
        } else {
            return url.split('/').pop();
        }
    }

    createPaeImage(pae, paeMax) {
        const dim = pae.length;
        const raw = new Uint8ClampedArray(dim * dim * 4);

        for (let y = 0; y < dim; y++) {
            for (let x = 0; x < dim; x++) {
                const coord = (y * dim + x) * 4;
                const value = pae[y][x];
                const color = this.#paePalette(value / paeMax)._rgb;

                for (const [i, colorValue] of color.slice(0, 3).entries()) {
                    raw[coord + i] = colorValue;
                }

                raw[coord + 3] = 255;
            }
        }

        return createImageBitmap(new ImageData(raw, dim)).then(bitmap => {
            const canvas = document.createElement('canvas');
            canvas.width = bitmap.width;
            canvas.height = bitmap.height;
            const ctx = canvas.getContext('bitmaprenderer');
            ctx.transferFromImageBitmap(bitmap);

            return new Promise(resolve => canvas.toBlob(resolve));
        }).then(URL.createObjectURL);
    }

    #initColors(subunits, colorMapping = null, colorRange = null) {
        if (colorMapping !== null) {
            for (const subunit of subunits.entries()) {
                subunit.color = colorMapping[subunit.uniprot];
            }
        } else if (colorRange !== null) {
            if (Array.isArray(colorRange)) {
                for (const [i, subunit] of subunits.entries()) {
                    subunit.color = colorRange[i % colorRange.length];
                }
            } else {
                for (const [i, subunit] of subunits.entries()) {
                    subunit.color = colorRange(i);
                }
            }
        }
    }

    #addDividers(sequenceLengths) {
        let offset = 0;

        for (const length of sequenceLengths.slice(0, -1)) {
            const ratio = this.#relative(length);

            for (const coord of ['x', 'y']) {

                const otherCoord = {'x': 'y', 'y': 'x'}[coord];
                const extent = Utils.toPercentage(offset + ratio);

                const line = Utils.createSVG('line', 'pv-dividers', {
                    stroke: this.#style.elements.dividers.color,
                    'stroke-width': this.#style.elements.dividers.thickness,
                    [coord + 1]: extent,
                    [coord + 2]: extent,
                    [otherCoord + 1]: '0',
                    [otherCoord + 2]: '100%',
                });
                this.#dividerGroup.appendChild(line);
            }

            offset += ratio;
        }
    }

    #addRegions(subunits) {
        const lengths = subunits.map(subunit => this.#relative(subunit.length));
        const offsets = [0, ...Utils.cumsum(lengths.slice(0, -1))];
        const indices = [...Array(subunits.length).keys()]
        const names = subunits.map(subunit => subunit.title);

        // create stripe patterns for individual regions
        const getPatternId =
            (i, j) => `stripes-${[i, j].sort().map(x => names[x]).join('-')}`;

        for (let i = 0; i < subunits.length; i++) {
            for (let j = i + 1; j < subunits.length; j++) {
                const stripes = this.#stripePattern.cloneNode(true);
                stripes.querySelector('rect').setAttribute(
                    'fill', subunits[i].color
                );
                stripes.querySelector('line').setAttribute(
                    'stroke', subunits[j].color
                );
                stripes.setAttribute('id', getPatternId(i, j));
                this.#graphDefs.appendChild(stripes);
            }
        }

        const regionSelections = new Map();
        const createRegionSelection = index => ({
            range: [
                [subunits[index].uniprot, 1],
                [subunits[index].uniprot, subunits[index].length]
            ],
            color: subunits[index].color
        });

        for (const [x, y] of Utils.cartesian(indices, indices)) {
            const region = Utils.createSVG('g', 'pv-region', {
                opacity: this.#style.elements.regions.opacity
            });
            const id = `${names[x]}-${names[y]}`;
            region.dataset.id = id;

            this.#regionGroup.appendChild(region);

            const background = Utils.createSVG('rect', null, {
                x: Utils.toPercentage(offsets[x]),
                y: Utils.toPercentage(offsets[y]),
                width: Utils.toPercentage(lengths[x]),
                height: Utils.toPercentage(lengths[y])
            });

            region.appendChild(background);

            const labelX = offsets[x] + lengths[x] / 2;
            const labelY = offsets[y] + lengths[y] / 2;
            const fontSize = this.#style.elements.regions.fontSize;

            const label = Utils.createSVG('text', null, {
                x: Utils.toPercentage(labelX),
                y: Utils.toPercentage(labelY),
                'text-anchor': 'middle',
                'dominant-baseline': 'middle',
                'font-size': fontSize * this.#viewBox.height,
                'font-family': this.#style.general.fontFamily,
            });

            if (x === y) {
                background.setAttribute('fill', subunits[x].color);
                label.textContent = names[x];
                region.appendChild(label);
                region.insertBefore(
                    this.#createBackgroundBox(label.getBBox(), 0.2 * fontSize),
                    label
                );

                const selection = createRegionSelection(x);

                let meanPae = null;

                if (this.#complex.pae) {
                    meanPae = this.#calcMeanPae(
                        ...selection.range,
                        ...selection.range
                    )
                }

                regionSelections.set(id, {
                    type: 'single',
                    selection: createRegionSelection(x),
                    meanPae: meanPae,
                });
            } else {
                background.setAttribute('fill', `url(#${getPatternId(x, y)})`);

                if (lengths[x] >= lengths[y]) {
                    label.textContent = `${names[x]} / ${names[y]}`;
                    region.appendChild(label);
                    region.insertBefore(
                        this.#createBackgroundBox(
                            label.getBBox(),
                            0.2 * fontSize
                        ),
                        label
                    );
                } else {
                    label.textContent = "/";

                    const upper = Utils.createSVG('text');
                    const lower = Utils.createSVG('text');

                    upper.textContent = names[x];
                    lower.textContent = names[y];

                    for (const text of [upper, lower]) {
                        Utils.setAttributes(text, {
                            x: Utils.toPercentage(labelX),
                            'text-anchor': 'middle',
                            'dominant-baseline': 'middle',
                            'font-size': fontSize * this.#viewBox.height,
                            'font-family': this.#style.general.fontFamily,
                        });
                    }

                    upper.setAttribute(
                        'y', Utils.toPercentage(labelY - fontSize)
                    );
                    lower.setAttribute(
                        'y', Utils.toPercentage(labelY + fontSize)
                    );

                    region.append(upper, label, lower);
                    const upperBox = upper.getBBox();
                    const lowerBox = lower.getBBox();

                    const combinedBox = {
                        x: Math.min(upperBox.x, upperBox.x),
                        y: upperBox.y,
                        width: Math.max(upperBox.width, upperBox.width),
                        height: lowerBox.y + lowerBox.height - upperBox.y,
                    };

                    region.insertBefore(
                        this.#createBackgroundBox(combinedBox, 0.2 * fontSize),
                        upper
                    );
                }

                const selectionX = createRegionSelection(x);
                const selectionY = createRegionSelection(y);

                let meanPae = null;

                if (this.#complex.pae) {
                    meanPae = this.#calcMeanPae(
                        ...selectionX.range,
                        ...selectionY.range
                    )
                }

                regionSelections.set(id, {
                    type: 'pair',
                    selection: {
                        x: selectionX,
                        y: selectionY,
                    },
                    meanPae: meanPae,
                });
            }
        }

        this.displayRegions(false);

        const toggleRegions = () => this.displayRegions(
            (this.#isShiftKeyDown && this.#isCursorOnGraph)
            || this.#regionToggleCheckBox.checked
        );

        document.addEventListener('keydown', event => {
            if (event.key === 'Shift') {
                this.#isShiftKeyDown = true;

                if (this.#isCursorOnGraph) {
                    this.#regionToggleCheckBox.checked = true;
                }
            }

            toggleRegions();
        });

        document.addEventListener('keyup', event => {
            if (event.key === 'Shift') {
                this.#isShiftKeyDown = false;
                if (this.#isCursorOnGraph) {
                    this.#regionToggleCheckBox.checked = false;
                }
            }

            toggleRegions();
        });

        document.addEventListener('mousemove', event => {
            this.#isCursorOnGraph = this.isCursorOnGraph(event);

            this.#updateCursorStatus(event);

            toggleRegions();
        });

        this.#regionToggleCheckBox.addEventListener('change', _ => {
            toggleRegions();
        });

        this.#regionGroup.addEventListener('click', event => {
            const region = event.target.closest('.pv-region');

            if (region === null) {
                return;
            }

            this.deselectAll(true);
            this.#selectedRegion?.classList.remove('pv-region-selected');
            this.#selectedRegion = event.target.closest('.pv-region');
            this.#selectedRegion.classList.add('pv-region-selected');

            const regionSelection = regionSelections.get(
                this.#selectedRegion.dataset.id
            );

            this.#updateRegionSelectionStatus(regionSelection);

            this.#element.dispatchEvent(
                new CustomEvent('pv-select-region', {
                    bubbles: true,
                    detail: {
                        complex: this.#complex,
                        ...regionSelection
                    }
                })
            );
        });

        this.#regionGroup.addEventListener('mousedown', event => {
            event.stopPropagation();
        });
    }

    #updateCursorStatus(event) {
        if (!this.#isCursorOnGraph) {
            this.#statusCursor.replaceChildren();
            return;
        }
        const coords = this.getRelativeMousePosition(
            event, false
        );

        const isCrosslink = event.target.classList.contains(
            'pv-crosslink-marker'
        );

        let residues;

        if (isCrosslink) {
            const crosslink = this.#crosslinks[
                event.target.dataset.crosslinkId
                ];

            residues = [
                [crosslink.get('Protein1'), parseInt(crosslink.get('SeqPos1'))],
                [crosslink.get('Protein2'), parseInt(crosslink.get('SeqPos2'))],
            ]
        } else {
            residues = coords.map(
                this.#getResidueFromRelative.bind(this)
            )
        }

        this.#statusCursor.replaceChildren(
            this.#getStatusAtCoords(residues[0], residues[1], isCrosslink)
        )
    }

    #getStatusAtCoords(coordX, coordY, isCrosslink) {
        const fragment = new DocumentFragment();

        const [residueX, residueY] = this.#getCoordStrings([coordX, coordY]);

        if (isCrosslink) {
            fragment.append(`Crosslink ${residueX.string} - ${residueY.string}`);

            if (this.#complex.pae) {
                const meanPae = (
                    this.#complex.pae[residueY.index][residueX.index]
                    + this.#complex.pae[residueX.index][residueY.index]
                ) / 2;
                const paeDisplay = document.createElement('b');
                paeDisplay.textContent
                    = `mean PAE: ${meanPae.toFixed(2)}`;
                fragment.append("; ", paeDisplay);
            }
        } else {
            const statusX = document.createElement('span');
            statusX.classList.add('pv-x');
            statusX.textContent = `X: ${residueX.string}`;

            const statusY = document.createElement('span');
            statusY.classList.add('pv-y');
            statusY.textContent = `Y: ${residueY.string}`;

            fragment.append(statusX, ", ", statusY);

            if (this.#complex.pae) {
                const pae = this.#complex.pae[residueY.index][residueX.index];
                const paeDisplay = document.createElement('b');
                paeDisplay.textContent = `PAE: ${pae.toFixed(2)}`;
                fragment.append("; ", paeDisplay);
            }
        }

        return fragment;
    }

    #getCoordStrings(residues) {
        return residues.map(([uniprot, coord]) => {
            const member = this.#complex.members.find(
                member => member.uniprot === uniprot
            );

            return {
                string: PaeViewer.residueToString(member, coord),
                index: member.offset + coord - 1
            }
        });
    }

    displayRegions(show) {
        this.#regionGroup.setAttribute(
            'visibility', show ? 'visible' : 'hidden'
        );
    }

    #getResidueFromRelative(coord) {
        coord = Math.ceil(coord * this.#dim);

        if (coord === 0) coord++;

        let uniprot = null;

        for (const member of this.#complex.members) {
            uniprot = member.uniprot;
            if (coord <= member.length) break;

            coord -= member.length;
        }

        return [uniprot, coord];
    }

    getRelativeMousePosition(mouseEvent, clamped = true) {
        const rect = this.#paeMatrix.getBoundingClientRect();

        const coords = [
            (mouseEvent.clientX - rect.left) / (rect.right - rect.left),
            (mouseEvent.clientY - rect.top) / (rect.bottom - rect.top)
        ];

        if (clamped) {
            return coords.map(coord => Utils.clamp(coord, 0, 1));
        } else {
            return coords;
        }
    }

    isCursorOnGraph(mouseEvent) {
        const coords = this.getRelativeMousePosition(mouseEvent, false);

        for (const coord of coords) {
            if (coord < 0 || coord > 1) {
                return false;
            }
        }

        return true;
    }

    /**
     * Updates the coordinates of the bounding lines of the selection
     * rectangle between point `from` (f) and `to`(t).
     * The lines will be drawn as follows:
     *
     *  y^               d     y^               d
     *   |              /       |              0
     *   |             /        |             /|
     *   |  t---O-----0         |      O-----0-t
     *   |  |   |    /          |      |    /  |
     *   |  O---f---0           |      f---0---O
     *   |  |   |  /            |      |  /
     *   |  |   | /             |      | /
     *   |  |   |/              |      |/
     *   |  |   0               |      0
     *   |  |  /                |     /
     *   |  | /                 |    /
     *   |  |/                  |   /
     *   |  0                   |  /
     *   | /                    | /
     *   |/                     |/
     *   +--------------->      +--------------->
     *                   x                      x
     * @param from
     * @param to
     */
    #updateSelectionLines(from, to) {
        for (const [index, [axis, constantPoint]] of [...Utils.cartesian(
            ['x', 'y'], [from, to]
        )].entries()) {
            const i = axis === 'x' ? 0 : 1;
            const j = 1 - i;
            const constantCoord = axis === 'x' ? 'y' : 'x';
            const constant = constantPoint[j];

            const line = this.#selection.lines[index];

            const [outer, inner] = [to[i], from[i]].sort((a, b) =>
                Math.abs(constant - b) - Math.abs(constant - a)
            );

            Utils.setAttributes(line, {
                [axis + 1]: Utils.toPercentage(outer),
                [axis + 2]: Utils.toPercentage(outer < constant ?
                    Math.max(inner, constant) : Math.min(inner, constant)
                ),
                [constantCoord + 1]: Utils.toPercentage(constant),
                [constantCoord + 2]: Utils.toPercentage(constant),
            });
        }
    }

    #updatedSelectionMarkers(from, to) {
        const inUpperHalf = (x, y) => y >= x;

        // start point is in upper left triangle
        const startInUpperHalf = inUpperHalf(from[0], from[1]);

        // if start is in one half and end point (or one of the other
        // points created by constructing the rectangle) is in the
        // other, there is an overlap
        const otherPointsInUpperHalf = [
            [to[0], to[1]], [to[0], from[1]], [from[0], to[1]]
        ].map(([x, y]) => inUpperHalf(x, y));

        const overlap = (
            (startInUpperHalf && !otherPointsInUpperHalf.every(Boolean))
            || (!startInUpperHalf && otherPointsInUpperHalf.some(Boolean))
        );

        const style = this.#style.elements.selection;
        const colors = [style.colors.x, style.colors.y];

        // make sure colors are correctly assigned to X and Y
        const [color1, color2] = from[1] >= from[0] ? colors : colors.reverse();

        const gradients = from[1] > from[0] ?
            ['rangeMarkerLower', 'rangeMarkerUpper']
            : ['rangeMarkerLowerReversed', 'rangeMarkerUpperReversed'];

        const markerColors = [
            color1,
            ...(overlap ?
                    gradients.map(gradient => `url("#${gradient}")`)
                    : [color1, color2]
            ),
            color2
        ];

        const markerCoords = [from[0], from[1], to[0], to[1]].sort();

        for (const [i, coord] of markerCoords.entries()) {
            const marker = this.#selection.rangeMarkers[i];
            Utils.setAttributes(marker, {
                cx: Utils.toPercentage(coord),
                cy: Utils.toPercentage(coord),
                fill: markerColors[i]
            });
        }

        const lineColors = [color1, style.colors.overlap, color2];

        for (const [i, [start, end]] of [
            ...Utils.pairwise(markerCoords)
        ].entries()) {
            const line = this.#selection.rangeLines[i];

            if (i === 1 && !overlap) {
                line.setAttribute('visibility', 'hidden');
                continue;
            } else {
                line.setAttribute('visibility', 'visible');
            }
            Utils.setAttributes(line, {
                x1: Utils.toPercentage(start),
                y1: Utils.toPercentage(start),
                x2: Utils.toPercentage(end),
                y2: Utils.toPercentage(end),
                stroke: lineColors[i]
            });
        }

        const coords = markerCoords.map(
            this.#getResidueFromRelative.bind(this)
        );

        const ranges = [coords.slice(0, 2), coords.slice(2, 4)];
        const [rangeX, rangeY] = startInUpperHalf ? ranges : ranges.reverse();

        const [fromResidueX, fromResidueY] = from.map(
            this.#getResidueFromRelative.bind(this)
        );

        const [toResidueX, toResidueY] = to.map(
            this.#getResidueFromRelative.bind(this)
        );

        return {
            x: {range: rangeX, color: style.colors.x},
            y: {range: rangeY, color: style.colors.y},
            overlap: overlap ? {
                range: coords.slice(1, 3),
                color: style.colors.overlap
            } : null,
            from: {x: fromResidueX, y: fromResidueY},
            to: {x: toResidueX, y: toResidueY},
        }
    }

    setupSelectionListeners() {
        let lastMouseDownEvent = null;
        let selectingRange = false;

        this.#interactiveGroup.addEventListener('mousedown', event => {
            this.deselectAll(false);

            event.stopPropagation();
            event.preventDefault();

            lastMouseDownEvent = event;
        });

        document.addEventListener('mouseup', event => {
            if (lastMouseDownEvent === null) return;

            event.stopPropagation();
            event.preventDefault();

            if (selectingRange) {
                this.selectRangeEnd(
                    this.getRelativeMousePosition(event),
                );

                selectingRange = false;
            } else {
                this.selectPoint(
                    this.getRelativeMousePosition(lastMouseDownEvent)
                );
            }

            lastMouseDownEvent = null;
        });

        document.addEventListener('mousemove', event => {
            if (lastMouseDownEvent === null) return;

            event.stopPropagation();
            event.preventDefault();

            if (selectingRange) {
                this.selectRangeUpdate(
                    this.getRelativeMousePosition(event)
                );
            } else {
                this.selectRangeStart(
                    this.getRelativeMousePosition(event),
                    this.getRelativeMousePosition(lastMouseDownEvent)
                );

                selectingRange = true;
            }

        });
    }

    selectPoint(point) {
        const [coordX, coordY] = point;
        const style = this.#style.elements.selection;

        const lineTemplate = Utils.createSVG('line', 'pv-selection-line', {
            stroke: style.lines.color,
            'stroke-width': style.lines.thickness,
            'stroke-dasharray': style.lines.dashLength,
            x2: Utils.toPercentage(coordX),
            y2: Utils.toPercentage(coordY)
        });

        for (const coord of point) {
            const line = lineTemplate.cloneNode(false);
            Utils.setAttributes(line, {
                x1: Utils.toPercentage(coord),
                y1: Utils.toPercentage(coord)
            });
            this.#selectionGroup.appendChild(line);
        }

        const markerTemplate = Utils.createSVG(
            'circle', 'pv-selection-marker', {
                stroke: style.markers.outlineColor,
                'stroke-width': style.markers.outlineThickness,
                r: style.markers.size
            });

        for (const [color, cx, cy] of [
            ['white', coordX, coordY],
            [style.colors.x, coordX, coordX],
            [style.colors.y, coordY, coordY],
        ]) {
            const marker = markerTemplate.cloneNode(false);
            Utils.setAttributes(marker, {
                cx: Utils.toPercentage(cx),
                cy: Utils.toPercentage(cy),
                fill: color
            });
            this.#selectionGroup.appendChild(marker);
        }

        const [selectionX, selectionY] = [coordX, coordY].map(
            this.#getResidueFromRelative.bind(this)
        );

        this.#statusSelection.replaceChildren(
            this.#getStatusAtCoords(selectionX, selectionY, false)
        );

        this.#element.dispatchEvent(new CustomEvent('pv-select-residue-pair', {
            bubbles: true,
            detail: {
                complex: this.#complex,
                selection: {
                    x: {residue: selectionX, color: style.colors.x},
                    y: {residue: selectionY, color: style.colors.y}
                }
            }
        }));
    }

    setRect(rect, x1, y1, x2, y2) {
        const [rectX1, rectX2] = [x1, x2].sort();
        const [rectY1, rectY2] = [y1, y2].sort();

        Utils.setAttributes(rect, {
            x: Utils.toPercentage(rectX1),
            y: Utils.toPercentage(rectY1),
            width: Utils.toPercentage(rectX2 - rectX1),
            height: Utils.toPercentage(rectY2 - rectY1),
        });
    }

    selectRangeStart(from, to) {
        const style = this.#style.elements.selection;
        this.#selection.start = from;

        const rect = Utils.createSVG('rect', 'pv-selection-rect', {
            fill: style.rect.color,
            opacity: style.rect.opacity,
            stroke: 'none'
        });
        this.setRect(rect, ...from, ...to);
        this.#selection.rect = rect;
        this.#selectionGroup.appendChild(rect);
        this.#selection.lines = [];

        const lineTemplate = Utils.createSVG('line', 'pv-selection-line', {
            stroke: style.lines.color,
            'stroke-width': style.lines.thickness,
            'stroke-dasharray': style.lines.dashLength
        });

        for (let i = 0; i < 4; i++) {
            const line = lineTemplate.cloneNode(false);
            this.#selectionGroup.appendChild(line);
            this.#selection.lines.push(line);
        }

        this.#updateSelectionLines(from, to);

        const markerTemplate = Utils.createSVG(
            'circle', 'pv-selection-marker', {
                fill: style.markers.outlineColor,
                stroke: style.markers.outlineColor,
                'stroke-width': style.markers.outlineThickness,
                r: style.markers.size
            });
        this.#selection.rangeMarkers = [];

        for (let i = 0; i < 4; i++) {
            const marker = markerTemplate.cloneNode(false);
            this.#selectionGroup.appendChild(marker);
            this.#selection.rangeMarkers.push(marker);
        }

        const rangeLineTemplate = Utils.createSVG(
            'line', 'pv-range-line', {'stroke-width': style.lines.thickness}
        );
        this.#selection.rangeLines = [];

        for (let i = 0; i < 3; i++) {
            const line = rangeLineTemplate.cloneNode(false);
            this.#selectionGroup.appendChild(line);
            this.#selection.rangeLines.push(line);
        }

        const rangeSelection = this.#updatedSelectionMarkers(from, to);

        const startMarker = markerTemplate.cloneNode(false);
        Utils.setAttributes(startMarker, {
            cx: Utils.toPercentage(from[0]),
            cy: Utils.toPercentage(from[1]),
        });
        this.#selectionGroup.appendChild(startMarker);

        this.#selection.rectMarkers = [];

        // rectangle markers
        for (const [cx, cy] of [
            [to[0], to[1]],
            [from[0], to[1]],
            [to[0], from[1]],
        ]) {
            const marker = markerTemplate.cloneNode(false);
            Utils.setAttributes(startMarker, {
                cx: Utils.toPercentage(cx),
                cy: Utils.toPercentage(cy),
            });
            this.#selectionGroup.appendChild(marker);
            this.#selection.rectMarkers.push(marker);
        }

        this.updateRangeSelectionStatus(rangeSelection);
    }

    updateRangeSelectionStatus(selection) {
        const residuesX = this.#getCoordStrings(selection.x.range);
        const residuesY = this.#getCoordStrings(selection.y.range);

        const statusX = document.createElement('span');
        statusX.classList.add('pv-x');
        statusX.textContent = `X: ${residuesX[0].string} - ${residuesX[1].string}`;

        const statusY = document.createElement('span');
        statusY.classList.add('pv-y');
        statusY.textContent = `Y: ${residuesY[0].string} - ${residuesY[1].string}`;

        this.#statusSelection.replaceChildren(statusX, ", ", statusY);

        if (selection.overlap !== null) {
            const residuesOverlap = this.#getCoordStrings(
                selection.overlap.range
            );

            const statusOverlap = document.createElement('span');
            statusOverlap.classList.add('pv-overlap');
            statusOverlap.textContent = `Overlap: ${residuesOverlap[0].string}`
                + ` - ${residuesOverlap[1].string}`;

            this.#statusSelection.append(", ", statusOverlap);
        }
    }

    #updateRegionSelectionStatus(region) {
        this.#statusSelection.replaceChildren();

        const createMarker = (selection) => {
            const marker = document.createElement('span');
            marker.classList.add('pv-color-marker');
            marker.style.backgroundColor = selection.color;

            const member = this.#complex.members.find(
                member => member.uniprot === selection.range[0][0]
            );

            marker.textContent = member.title;
            return marker;
        };

        if (region.type === 'single') {
            this.#statusSelection.append(
                "chain",
                createMarker(region.selection)
            );
        } else {
            const [chainX, chainY] = [
                region.selection.x, region.selection.y
            ].map(createMarker);

            this.#statusSelection.append("chains ", chainX, " / ", chainY);
        }

        if (region.meanPae !== null) {
            const paeDisplay = document.createElement('b');
            paeDisplay.textContent = `mean PAE: ${region.meanPae.toFixed(2)}`;

            this.#statusSelection.append("; ", paeDisplay);
        }
    }

    selectRangeUpdate(to) {
        const from = this.#selection.start;

        this.setRect(this.#selection.rect, ...from, ...to);

        // rectangle markers
        for (const [i, [cx, cy]] of [
            [to[0], to[1]],
            [from[0], to[1]],
            [to[0], from[1]],
        ].entries()) {
            const marker = this.#selection.rectMarkers[i];
            Utils.setAttributes(marker, {
                'cx': Utils.toPercentage(cx),
                'cy': Utils.toPercentage(cy)
            });
        }

        this.#updateSelectionLines(from, to);

        const rangeSelection = this.#updatedSelectionMarkers(from, to);
        this.updateRangeSelectionStatus(rangeSelection);

        return rangeSelection;
    }

    selectRangeEnd(to) {
        const selection = this.selectRangeUpdate(to);

        this.#element.dispatchEvent(new CustomEvent('pv-select-residue-range', {
            bubbles: true,
            detail: {
                complex: this.#complex,
                selection: selection,
            }
        }));

        if (this.#complex.pae) {
            // calculate mean of 2D slice of PAE matrix
            const meanPae = this.#calcMeanPae(
                selection.from.x,
                selection.to.x,
                selection.from.y,
                selection.to.y
            );

            const paeDisplay = document.createElement('b');
            paeDisplay.textContent = `mean PAE: ${meanPae.toFixed(2)}`;

            this.#statusSelection.append("; ", paeDisplay);
        }
    }

    #calcMeanPae(residueX1, residueX2, residueY1, residueY2) {
        const sortNumerically = (a, b) => a - b;

        const [x1, x2] = this.#getCoordStrings(
            [residueX1, residueX2]
        ).map(residue => residue.index).sort(sortNumerically);

        const [y1, y2] = this.#getCoordStrings(
            [residueY1, residueY2]
        ).map(residue => residue.index).sort(sortNumerically);

        // calculate mean of 2D slice of PAE matrix
        return Utils.mean(
            this.#complex.pae.slice(y1, y2 + 1).map(
                row => row.slice(x1, x2 + 1)
            ).flat()
        );
    }

    #addTick(
        rootTick, rootLabel, axis, value, offset, tickLength, text = null
    ) {
        const otherAxis = axis === 'x' ? 'y' : 'x';
        const anchor = axis === 'x' ? 'middle' : 'end';
        const baseline = axis === 'x' ? 'auto' : 'central';

        const ratio = this.#relative(value + offset);

        const tick = Utils.createSVG('line', null, {
            stroke: this.#style.elements.ticks.color,
            'stroke-width': this.#style.elements.ticks.thickness,
            [axis + 1]: '0',
            [axis + 2]: Utils.toPercentage(-tickLength),
            [otherAxis + 1]: Utils.toPercentage(ratio),
            [otherAxis + 2]: Utils.toPercentage(ratio),
        });

        const labelCoords = [
            ratio,
            -(tickLength + this.#style.elements.ticks.labelGap)
        ];

        rootTick.appendChild(tick);
        this.#addTickLabel(
            rootLabel, ...(axis === 'x' ? labelCoords : labelCoords.reverse()),
            text !== null ? text : value, anchor, baseline
        );
    }

    #addTicks(members) {
        let offset = 0;
        let lastSubunit = null;
        const style = this.#style.elements;
        const interval = style.ticks.unitInterval;

        this.#addTickLabel(
            this.#axesGroup, -style.ticks.labelGap, -style.ticks.labelGap,
            "0", 'end', 'auto', false
        );

        // add tick marks for both axes
        for (const [i, member] of members.entries()) {
            for (const axis of ['x', 'y']) {
                // add value ticks as multiples of interval
                for (
                    let value = interval;
                    // prevents overlap
                    value < member.length - 0.1 * interval;
                    value += interval
                ) {
                    this.#addTick(
                        this.#unitTicksGroup, this.#unitTickLabelsGroup,
                        axis, value, offset, style.ticks.units.length
                    );
                }

                // add complex subunit ticks
                this.#addTick(
                    this.#sequenceTicksGroup, this.#sequenceTickLabelsGroup,
                    axis, member.length, offset, style.ticks.subunits.length,
                    member.length + (i < members.length - 1 ? ' / 0' : '')
                );

                const labelCoords = [
                    this.#relative(offset + member.length / 2),
                    -style.subunitLabels.gap
                ];

                this.#addTickLabel(
                    this.#sequenceTickLabelsGroup,
                    ...(axis === 'x' ? labelCoords : labelCoords.reverse()),
                    member.title,
                    ...(axis === 'x' ? ['middle', 'auto'] : ['end', 'central']),
                    true,
                    {
                        'font-weight': style.subunitLabels.fontWeight,
                        'font-style': style.subunitLabels.fontStyle,
                        'fill': style.subunitLabels.color
                    },
                    {'fill': members[i].color}
                )
            }

            offset += member.length;
            lastSubunit = member.length;
        }
    }

    #createBackgroundBox(
        box, horizontalPadding = 0, verticalPadding = 0, params = {}
    ) {
        horizontalPadding = horizontalPadding * this.#viewBox.width;
        verticalPadding = verticalPadding * this.#viewBox.height;

        return Utils.createSVG('rect', null, {
            rx: this.#style.elements.boxes.roundness,
            fill: this.#style.elements.boxes.color,
            opacity: this.#style.elements.boxes.opacity,
            x: Utils.toPercentage(
                (box.x - horizontalPadding) / this.#viewBox.width
            ),
            y: Utils.toPercentage(
                (box.y - verticalPadding) / this.#viewBox.height
            ),
            width: Utils.toPercentage(
                (box.width + 2 * horizontalPadding) / this.#viewBox.width
            ),
            height: Utils.toPercentage(
                (box.height + 2 * verticalPadding) / this.#viewBox.height
            ),
            ...params
        });
    }

    #addTickLabel(
        root, x, y, text, anchor, baseline, addBackground = true,
        params = {}, backgroundParams = {}
    ) {
        const fontSize = this.#style.elements.ticks.fontSize;

        const label = Utils.createSVG('text', 'pv-tick-label', {
            x: Utils.toPercentage(x),
            y: Utils.toPercentage(y),
            'text-anchor': anchor,
            'dominant-baseline': baseline,
            'font-size': fontSize * this.#viewBox.height,
            'font-family': this.#style.general.fontFamily,
            ...params
        });
        label.textContent = text;
        root.appendChild(label);

        if (addBackground) {
            const background = this.#createBackgroundBox(
                label.getBBox(), 0.1 * fontSize, 0, backgroundParams
            );
            root.insertBefore(background, label);
        }
    }

    addCrosslinks(crosslinks, members) {
        this.#crosslinks = crosslinks;
        let offset = 0;

        for (const member of members) {
            member['offset'] = offset;
            offset += member.length;
        }

        members = new Map(members.map(member => [member.uniprot, member]));
        const style = this.#style.elements.crosslinks;

        this.#crosslinkMarkers = [];

        for (const [id, crosslink] of crosslinks.entries()) {
            const restraintSatisfied = crosslink.has('RestraintSatisfied') ?
                crosslink.get('RestraintSatisfied').toLowerCase() === 'true'
                : true;

            const color = restraintSatisfied ?
                style.restraintColors.satisfied
                : style.restraintColors.violated;

            const coords = [1, 2].map(i => {
                const uniprot = crosslink.get('Protein' + i);
                return this.#relative(
                    members.get(uniprot)['offset']
                    + parseInt(crosslink.get('SeqPos' + i))
                );
            });

            const markerPair = [];

            for (const [coord1, coord2] of [coords, [...coords].reverse()]) {
                const crosslinkMarker = Utils.createSVG(
                    'circle', 'pv-crosslink-marker', {
                        r: style.size,
                        stroke: style.outlineColor,
                        'stroke-width': style.outlineThickness,
                        opacity: style.opacity,
                        'data-crosslink-id': id,
                        cx: Utils.toPercentage(coord1),
                        cy: Utils.toPercentage(coord2),
                        fill: color
                    });

                this.#crosslinkGroup.appendChild(crosslinkMarker);

                markerPair.push(crosslinkMarker);
            }

            this.#crosslinkMarkers.push(markerPair);
        }

        this.#crosslinkGroup.addEventListener('mousedown', event => {
            event.stopPropagation();
        });

        this.#crosslinkGroup.addEventListener('click', event => {
            if (!event.target.classList.contains('pv-crosslink-marker')) {
                return;
            }

            event.stopPropagation();
            const id = event.target.dataset.crosslinkId;
            this.deselectAll(true);

            for (const markerPair of this.#crosslinkMarkers) {
                for (const marker of markerPair) {
                    marker.classList.add('pv-crosslink-unselected');
                    marker.classList.remove('pv-crosslink-selected');
                }
            }

            for (const marker of this.#crosslinkMarkers[id]) {
                marker.classList.remove('pv-crosslink-unselected');
                marker.classList.add('pv-crosslink-selected');
            }

            const crosslink = this.#crosslinks[id];

            const [residue1, residue2] = [
                [crosslink.get('Protein1'), parseInt(crosslink.get('SeqPos1'))],
                [crosslink.get('Protein2'), parseInt(crosslink.get('SeqPos2'))],
            ];

            this.#statusSelection.replaceChildren(
                this.#getStatusAtCoords(residue1, residue2, true)
            );

            this.#element.dispatchEvent(new CustomEvent('pv-select-crosslink', {
                bubbles: true,
                detail: {
                    complex: this.#complex,
                    selection: {
                        id: event.target.dataset.crosslinkId,
                        residue1: residue1,
                        residue2: residue2,
                    }
                }
            }));
        });

        this.#crosslinkGroup.addEventListener('mouseover', event => {
            const id = event.target.dataset.crosslinkId;

            for (const marker of this.#crosslinkMarkers[id]) {
                marker.classList.add('pv-crosslink-hover');
            }
        });

        this.#crosslinkGroup.addEventListener('mouseout', event => {
            const id = event.target.dataset.crosslinkId;

            for (const marker of this.#crosslinkMarkers[id]) {
                marker.classList.remove('pv-crosslink-hover');
            }
        });
    }

    #dispatchSelectionReset() {
        this.#element.dispatchEvent(new CustomEvent('pv-reset-selection', {
            bubbles: true,
            detail: {
                complex: this.#complex
            }
        }));
    }

    deselectAll(dispatchEvent = true) {
        if (dispatchEvent) {
            this.#dispatchSelectionReset();
        }

        for (const markerPair of this.#crosslinkMarkers) {
            for (const marker of markerPair) {
                marker.classList.remove('pv-crosslink-unselected');
                marker.classList.remove('pv-crosslink-selected');
            }
        }

        this.#resetSelection();

        this.#selectedRegion?.classList.remove('pv-region-selected');
        this.#selectedRegion = null;

        this.#selectionGroup.replaceChildren();
        this.#statusSelection.replaceChildren();
    }

    static residueToString(member, index) {
        const residue = member.sequence[index - 1];

        return `${residue.name} ${index} (${member.title})`;
    }
}
