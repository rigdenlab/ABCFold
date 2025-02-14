import {PaeViewer} from "./pae-viewer.js";
import {StructureCarousel} from "./structure-carousel.js";
import {ComplexSequenceViewer} from "./complex-sequence-viewer.js";
import * as Utils from "./utils.js";
import {createRandomId} from "./utils.js";
import {readCrosslinks} from "./read-crosslinks.js";
import {readScores} from "./read-scores.js";
import {readStructure} from "./read-structure.js";

new bootstrap.Modal(document.getElementById('cookieConsent')).show();

const toast = document.querySelector('.toast-pae-max-warning');
const maxPaeWarning = new bootstrap.Toast(toast);
const maxPaeWarningMessage = toast.querySelector('.toast-body');

const modificationWarning = new bootstrap.Toast(
    document.querySelector('.toast-modification-warning')
);

let paeViewer = null;
const paeViewerRoot = document.querySelector('#pae-viewer');
const paeViewerTemplate = document.querySelector('#pae-viewer-template');

let complexSequenceViewer = null;
const complexSequenceViewerRoot = document.querySelector(
    '#complex-sequence-viewer'
);
const complexSequenceViewerTemplate = document.querySelector(
    '#complex-sequence-viewer-template'
);

let structureCarousel = null;
const structureCarouselRoot = document.querySelector('#test-carousel');
const structureCarouselTemplate = document.querySelector(
    '#structure-carousel-template'
);
const structureViewerTemplate = document.querySelector(
    '#structure-viewer-template'
);
const structurePanelTemplate = document.querySelector(
    '#structure-panel-demo-template'
);

const chooseExampleFields = document.querySelector('#chooseExampleFields');
const uploadStructureSubmit = document.querySelector('#uploadStructureSubmit');

let structureViewerDisplay = null;

document.addEventListener('mousedown', event => {
    if (!structureViewerDisplay) {
        return;
    }

    if (!structureViewerDisplay.item.contains(event.target)) {
        paeViewer?.deselectAll(true);
    }
});

function loadComplexById(id) {
    chooseExampleFields.disabled = true;
    uploadStructureSubmit.disabled = true;

    fetch(
        `predictedComplex?id=${id}`,
        {'headers': {'Accept': 'application/json'}}
    ).then(response => response.json()).then(complex => {
        paeViewer = new PaeViewer(paeViewerRoot, paeViewerTemplate);

        const directory = `crosslinks/${complex.handle}/`;
        complex.paeImageUrl = directory + complex.handle + '.png';
        paeViewer.load(complex);

        complexSequenceViewer = new ComplexSequenceViewer(
            complexSequenceViewerRoot,
            complexSequenceViewerTemplate,
            complex
        );

        structureCarousel = new StructureCarousel(
            structureCarouselRoot,
            structureCarouselTemplate,
            structureViewerTemplate,
            structurePanelTemplate
        );

        structureCarousel.addStructure({
            id: id,
            type: "Predicted Complex",
            source: "predicted-complex",
            description: "Complex predicted with AlphaFold-Multimer."
        }, true).then(entry => {
            structureViewerDisplay = entry;
            chooseExampleFields.disabled = false;
            uploadStructureSubmit.disabled = false;
        });
    });
}

function loadComplex(complex, crosslinks) {
    resetAll();
    chooseExampleFields.disabled = true;
    uploadStructureSubmit.disabled = true;

    paeViewer = new PaeViewer(paeViewerRoot, paeViewerTemplate);

    complex.crosslinks = crosslinks;
    paeViewer.load(complex);

    complexSequenceViewer = new ComplexSequenceViewer(
        complexSequenceViewerRoot, complexSequenceViewerTemplate, complex
    );

    structureCarousel = new StructureCarousel(
        structureCarouselRoot,
        structureCarouselTemplate,
        structureViewerTemplate,
        structurePanelTemplate
    );

    const meta = {
        id: complex.id,
        source: "predicted-complex",
        type: "Predicted Complex",
        url: complex.structureFile,
        description: "User uploaded structure.",
        complex: complex,
        crosslinks: crosslinks
    };

    structureViewerDisplay = structureCarousel.addStructureMeta(meta, true);
    chooseExampleFields.disabled = false;
    uploadStructureSubmit.disabled = false;
}

function resetAll() {
    paeViewer = null;
    complexSequenceViewer = null;
    structureCarousel = null;

    paeViewerRoot.replaceChildren();
    complexSequenceViewerRoot.replaceChildren();
    structureCarouselRoot.replaceChildren();

    structureViewerDisplay = null;
}

loadComplexById(161);

document.querySelector('#chooseExample').addEventListener('submit', event => {
    event.preventDefault();

    resetAll();
    loadComplexById(event.submitter.value);
});

const uploadForm = document.querySelector('#uploadStructure');
const chainLabelsInput = document.querySelector('#chainLabels');
const chainLabelsFeedback = document.querySelector('#chainLabelsFeedback');
const structureFileInput = document.querySelector('#structureFile');
const structureFileFeedback = document.querySelector('#structureFileFeedback');
const scoresFileInput = document.querySelector('#scoresFile');
const scoresFileFeedback = document.querySelector('#scoresFileFeedback');
const crosslinksFileInput = document.querySelector('#crosslinksFile');
const crosslinksFileFeedback = document.querySelector('#crosslinksFileFeedback');

uploadForm.addEventListener('submit', event => {
    event.preventDefault();

    const data = new FormData(event.target);

    let chainLabels = data.get('chainLabels').trim();

    chainLabels = chainLabels ?
        chainLabels.split(';').map(label => label.trim()) : null;

    if (chainLabels?.some(label => label.length === 0)) {
        chainLabelsInput.setCustomValidity("label-empty");
        chainLabelsFeedback.textContent
            = "Labels must be non-empty.";
    } else if (chainLabels !== null
        && (new Set(chainLabels)).size !== chainLabels.length) {
        chainLabelsInput.setCustomValidity("label-non-unique");
        chainLabelsFeedback.textContent
            = "Labels must be unique.";
    } else {
        chainLabelsInput.setCustomValidity("");
    }

    structureFileFeedback.textContent = "";
    scoresFileFeedback.textContent = "";
    crosslinksFileInput.textContent = "";

    if (data.get('structureFile').name.length > 0) {
        structureFileInput.setCustomValidity('');
    }

    if (data.get('scoresFile').name.length > 0) {
        scoresFileInput.setCustomValidity('');
    }

    if (data.get('crosslinksFile').name.length > 0) {
        crosslinksFileInput.setCustomValidity('');
    }

    if (!uploadForm.checkValidity()) {
        uploadForm.classList.add('was-validated');
        return;
    }


    readStructure(data.get('structureFile'), chainLabels).then(async (structure) => {
        return [structure, await readScores(data.get('scoresFile'), structure.modifications)];
    }).then(([structure, scores]) => {
        const chains = structure.chains;
        uploadForm.classList.remove('was-validated');

        const totalLength = Utils.sum(
            chains.map(subunit => subunit.length)
        );

        if (scores.pae.length !== totalLength) {
            throw {
                type: 'scores-input-error',
                message: `The number of residues in the PAE matrix`
                    + ` (${scores.pae.length}) does not match the number of`
                    + ` residues in the structure (${totalLength})!`,
            };
        }

        if (scores.parsedMaxPae && scores.foundMaxPae > scores.parsedMaxPae) {
            maxPaeWarningMessage.textContent = (
                `A maximum PAE of  ${scores.parsedMaxPae} was supplied in the`
                + `scores file,  however, the highest PAE found was`
                + ` ${scores.foundMaxPae}. The maximum PAE was set to the`
                + ` latter value for scaling.`
            );
            maxPaeWarning.show();
        } else {
            maxPaeWarning.hide();
        }

        if (structure.modifications.length > 0) {
            modificationWarning.show();
        } else {
            modificationWarning.hide();
        }

        const subunitNames = chains.map(subunit => subunit.id);

        const complex = {
            id: createRandomId(),
            handle: subunitNames.join('-'),
            title: subunitNames.join('-'),
            chains: Object.fromEntries(chains.map(
                chain => [chain.id, chain.chain]
            )),
            members: chains,
            crosslinksFile: null,
            crosslinksUrl: data.get('crosslinksFile').name.length > 0 ?
                data.get('crosslinksFile') : null,
            structureFile: data.get('structureFile'),
            plddt: scores.meanPlddt,
            ptm: scores.ptm,
            iptm: scores.iptm,
            pae: scores.pae,
            maxPae: Math.max(scores.parsedMaxPae, scores.foundMaxPae)
        };

        return readCrosslinks(complex);
    }).then(
        ([complex, crosslinks]) => {
            uploadForm.reset();
            loadComplex(complex, crosslinks);
            complexSequenceViewerRoot.scrollIntoView();
        }
    ).catch(error => {
        console.error(error);

        const fields = new Map([
            ['invalid-chain-labels', {
                input: chainLabelsInput, feedback: chainLabelsFeedback
            }],
            ['structure-input-error', {
                input: structureFileInput, feedback: structureFileFeedback
            }],
            ['scores-input-error', {
                input: scoresFileInput, feedback: scoresFileFeedback
            }],
            ['crosslinks-input-error', {
                input: crosslinksFileInput, feedback: crosslinksFileFeedback
            }],
        ]);

        const field = fields.get(error?.type);

        if (field) {
            field.input.setCustomValidity('input-error');
            field.feedback.textContent = error.message;
            uploadForm.classList.add('was-validated');
        } else {
            throw error;
        }
    });
});
