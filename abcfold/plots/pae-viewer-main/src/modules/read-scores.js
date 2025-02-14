import {mean} from "./utils.js";

function isNumber(value) {
    return typeof value === 'number' && !isNaN(value);
}

export function readScores(file, modifications) {
    const reader = new FileReader();

    const promise = new Promise((resolve, reject) => {
        reader.addEventListener('load', () => {
            try {
                const parsed = parseScores(reader.result);
                const pae = processPae(parsed.pae, modifications);

                let foundMax = 0;

                for (const row of pae) {
                    for (const value of row) {
                        if (value > foundMax) {
                            foundMax = value;
                        }
                    }
                }

                resolve({
                    pae: pae,
                    parsedMaxPae: parsed.maxPae,
                    foundMaxPae: foundMax,
                    plddt: parsed.plddt,
                    meanPlddt: parsed.plddt ? mean(parsed.plddt) : null,
                    ptm: parsed.ptm,
                    iptm: parsed.iptm,
                });
            } catch (e) {
                reject(e);
            }
        }, false);

        reader.addEventListener('error', () => {
            reject(error("The JSON file couldn't be read correctly!"));
        });
    });

    reader.readAsText(file);
    return promise;
}

function error(message) {
    return {
        type: 'scores-input-error',
        message: message,
    }
}

function parseScores(text) {
    let data = null;

    try {
        data = JSON.parse(text);
    } catch (e) {
        throw error("The JSON file couldn't be parsed correctly!");
    }

    if (Array.isArray(data)) {
        if (data.length === 0) {
            throw error("The JSON file was empty!");
        }

        data = data[0];
    }

    if (data === null || typeof data !== 'object' || Array.isArray(data)) {
        throw error(
            "The JSON file must contain either an object or"
            + " an array where the first element is an object!",
        );
    }

    const maxPae = data['max_pae'] ?? data['max_predicted_aligned_error'] ?? null;

    if (maxPae !== null) {
        if (!isNumber(maxPae) || maxPae <= 0) {
            throw error(
                `The maximum PAE must be a number > 0, was`
                + ` '${JSON.stringify(maxPae)}'!`);
        }
    }

    const getNumeric = (data, key, label) => {
        const value = data[key] ?? null;

        if (value !== null && (!isNumber(value) || value < 0 || value > 1)) {
            throw error(
                `The ${label} must be a number in [0,1], but was `
                + ` '${JSON.stringify(value)}'!`);
        }

        return value;
    };

    return {
        pae: getPae(data),
        maxPae: maxPae,
        plddt: getPlddt(data),
        ptm: getNumeric(data, 'ptm', 'pTM'),
        iptm: getNumeric(data, 'iptm', 'ipTM'),
    }
}

/** Gets PAE matrix from data and validates it. */
function getPae(data) {
    const pae = data['pae'] ?? data['predicted_aligned_error'];
    let length = 0;

    if (pae === undefined) {
        throw error(
            "The JSON file must contain a key 'pae'/'predicted_aligned_error'!"
        );
    }

    if (!Array.isArray(pae)) {
        throw error("The PAE value must be an N*N array of numbers!");
    }

    length = pae.length;

    if (length === 0) {
        throw error("The PAE array can't be empty!");
    }

    for (const [rowIndex, row] of pae.entries()) {
        if (!Array.isArray(row)) {
            throw error(
                `The PAE value must be an N*N array of numbers! However, row`
                + ` ${rowIndex} is not an array!`,
            );
        }

        if (row.length !== length) {
            throw error(
                `The PAE value must be an N*N array of  numbers! However, the`
                + ` number of columns (${row.length}) in row ${rowIndex} does`
                + ` not match number of rows (${length})!`,
            );
        }

        for (const [columnIndex, value] of row.entries()) {
            if (!isNumber(value) || value < 0) {
                throw error(
                    `The PAE values must be numbers >= 0, but column ` +
                    ` ${columnIndex} in row ${rowIndex} contained`
                    + ` '${JSON.stringify(value)}'!`,
                );
            }
        }
    }

    return pae;
}

/** Gets pLDDT array from data and validates it. */
function getPlddt(data, length) {
    const plddt = data['plddt'] ?? null;

    if (plddt !== null) {
        if (!Array.isArray(plddt)) {
            throw error(
                "The pLDDT value must be an array of numbers"
                + " with the number of residues as length!",
            );
        } else if (plddt.length !== length) {
            throw error(
                `Length of pLDDT array (${plddt.length})`
                + ` didn't match length of PAE array (${length})!`,
            );
        } else {
            for (const [columnIndex, value] of plddt.entries()) {
                if (!isNumber(value) || value < 0 || value > 100) {
                    throw error(
                        `The pLDDT values must be numbers between 0 and 100,`
                        + ` but column ${columnIndex} contained`
                        + ` '${JSON.stringify(value)}'!`,
                    );
                }
            }
        }
    }

    return plddt;
}

/**
 * Replace atom-wise PAE values of modified residues with residue-wise
 * values, using the arithmetic means of the atom-wise values.
 */
function processPae(pae, modifications) {
    const matrix = pae.map(row => row.slice());

    // set rows at modification indices to mean of corresponding rows/columns
    for (const {index, atomCount} of modifications) {
        const meanByColumn = getRowMeanByColumn(matrix.slice(index, index + atomCount));

        for (let i = 0; i < atomCount; i++) {
            matrix[index + i] = meanByColumn;
        }

        for (let row of matrix) {
            const meanByRow = mean(row.slice(index, index + atomCount));

            for (let i = 0; i < atomCount; i++) {
                row[index + i] = meanByRow;
            }
        }
    }

    // set all but one value per row/column corresponding to replaced
    // atom PAE values with null to get filtered later
    // start with 1 to skip rows/columns with calculated mean values
    for (const {index, atomCount} of modifications) {
        for (let i = 1; i < atomCount; i++) {
            matrix[index + i] = null;
            matrix.forEach(row => {
                if (row) {
                    row[index + i] = null;
                }
            });
        }
    }

    return matrix.filter(row => row !== null).map(
        row => row.filter(value => value !== null)
    );
}

/**
 * Calculate the arithmetic mean by column over multiple rows.
 * Example:
 * Input: [
 *   [-1,  49, -120],
 *   [ 1,  50, -130],
 *   [ 0,  51, -50],
 * ]
 *
 * Output: [0, 50, -100]
 *
 */
function getRowMeanByColumn(rows) {
    return rows[0].map((_, i) => mean(rows.map(row => row[i])));
}
