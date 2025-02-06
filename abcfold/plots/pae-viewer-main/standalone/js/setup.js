
// https://stackoverflow.com/a/73891404


async function replaceAsync(string, regexp, replacerFunction) {
    const replacements = await Promise.all(
        Array.from(string.matchAll(regexp), replacerFunction));
    let i = 0;

    return string.replace(regexp, () => replacements[i++]);
}

function compileTemplate(template, relativePath) {
    return replaceAsync(template, /\{\{(\S+)}}/g, async (match) => {
        const value = match[1];

        if (value.startsWith('@')) {
            console.log(`${relativePath}/src/${value.slice(1)}`);
            return Promise.resolve(`${relativePath}/src/${value.slice(1)}`);
        } else {
            return await fetch(`${relativePath}/src/templates/${value}`)
                .then(response => response.text())
                .then(template => {
                    return compileTemplate(template, relativePath);
                });
        }
    });
}

const main = document.querySelector('main');
document.addEventListener("DOMContentLoaded", function() {
    const relativePath = document.getElementById('relativePath').value;

    compileTemplate(main.innerHTML, relativePath).then((result) => {
        main.innerHTML = result;

        main.querySelector('button[data-bs-target="#examples"]').parentElement.remove();

        import(`./pae-viewer-standalone.js`);
    });
});
