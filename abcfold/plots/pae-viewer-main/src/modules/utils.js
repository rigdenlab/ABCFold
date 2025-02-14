export function createRandomId(prefix = '') {
    if (prefix) {
        prefix += '-';
    }

    return prefix + Math.random().toString(36).slice(2);
}

export function positiveModulo(x, n) {
    return ((x % n) + n) % n;
}

export function splitLines(text) {
    return text.split(/\r?\n/);
}

/**
 * Create table (arrays of rows as Maps) from DSV
 * (Delimiter-Separated Values) string (multiple lines).
 *
 * @param dsv string in DSV format
 * @param headers array of column names; if null, treat first row as
 *                column names
 * @param sep separator character
 * @returns {*}
 */
export function readDSV(dsv, headers = null, sep = '\t') {
    let rows = splitLines(dsv)
        .filter(line => line !== "")
        .map(line => line.split(sep));

    if (headers === null) {
        headers = rows[0];
        rows = rows.slice(1);
    }

    for (const [i, row] of rows.entries()) {
        if (row.length !== headers.length) {
            throw {
                name: 'MalformattedDsvRow',
                message:
                    `Malformatted DSV: while parsing a DSV with`
                    + ` ${headers.length} headers (${headers.join(',')}), row`
                    + ` {i} (${row.join(',')}) had wrong number of fields`
                    + ` (${row.length})!`,
                headers: headers,
                separator: sep,
                row: row,
                index: i
            };
        }
    }

    return rows.map(row => new Map(row.map(
        (value, i) => [headers[i], value])
    ));
}

export async function fetchDSV(url, columns = null, sep = '\t') {
    return fetch(url).then(response => response.text()).then(text =>
        readDSV(text, columns, sep)
    );
}

/**
 * Partition an iterable based on a condition.
 *
 * @param iter iterable
 * @param doesConditionHold callable taking single argument and
 *                          outputting boolean
 * @returns {*} array of two arrays, the first one containing the
 *              elements for which the condition holds, the second one
 *              containing the rest
 */
// based on https://codereview.stackexchange.com/a/162879
export function partitionOn(iter, doesConditionHold) {
    return iter.reduce((result, element) => {
        result[doesConditionHold(element) ? 0 : 1].push(element);
        return result;
    }, [[], []]);
}

export function createSVG(type, cssClass = null, attributes = {}) {
    const el = document.createElementNS('http://www.w3.org/2000/svg', type);

    if (cssClass !== null) {
        if (typeof cssClass === 'string') {
            el.classList.add(cssClass);
        } else {  // multiple classes
            el.classList.add(...cssClass);
        }
    }
    setAttributes(el, attributes);

    return el;
}

export function setAttributes(element, params) {
    for (const [attribute, value] of Object.entries(params)) {
        element.setAttribute(attribute, value);
    }
}

export function toPercentage(value) {
    return `${value * 100}%`
}

// https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/
//  Global_Objects/Array/Reduce#sum_of_values_in_an_object_array
export function sum(array) {
    return array.reduce((previous, current) => previous + current, 0);
}

/** Return arithmetic mean of values. */
export function mean(array) {
    return array.reduce((a, b) => a + b, 0) / array.length;
}

export function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
}

/**
 * Iterates over pairs of elements of iterable.
 *
 * Source: https://stackoverflow.com/a/54458643
 *
 * @param iterable
 * @returns {Generator<unknown[], void, *>}
 */
export function* pairwise (iterable) {
    const iterator = iterable[Symbol.iterator]();
    let current = iterator.next();
    let next = iterator.next();
    while (!next.done) {
        yield [current.value, next.value];
        current = next;
        next = iterator.next();
    }
}

/**
 * Returns generator for the cartesian product of multiple
 * arrays.
 *
 * Source: https://stackoverflow.com/a/44012184
 *
 * @param head
 * @param tail
 * @returns {Generator<*[], void, *>}
 */
export function* cartesian(head, ...tail) {
    const remainder = tail.length > 0 ? cartesian(...tail) : [[]];
    for (let r of remainder) for (let h of head) yield [h, ...r];
}

/**
 * Creates bicolored, striped CSS gradient.
 *
 * @param {string} colorHex1
 * @param {string}  colorHex2
 * @param {number} opacity
 * @param {number} width
 * @returns {string}
 */
export function createStripes(colorHex1, colorHex2, opacity, width = 0.3) {
    const opacityHex = (Math.round(opacity * 255)).toString(16);
    const rgba1 = colorHex1 + opacityHex;
    const rgba2 = colorHex2 + opacityHex;

    return `repeating-linear-gradient(`
        + `-45deg, ${rgba1}, ${rgba1} ${width}em,`
        + ` ${rgba2} ${width}em, ${rgba2} ${2 * width}em`
        + `)`;
}


export function cumsum(values) {
    let sum = 0;
    return values.map(value => sum += value);
}

export function readAsText(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsText(file);
    });
}

export function getFileExtension(file) {
    return file.name.split('.').pop();
}


export function* range(start, end, step = 1) {
    for (let i = start; i < end; i += step) {
        yield i;
    }
}
