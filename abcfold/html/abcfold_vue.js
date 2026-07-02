/*
data attribute creates object that is bound to this

computed - computed property but with caching
method - computed property - invoked each time

v-model - links inputs to vue js data


https://codepen.io/pespantelis/pen/ojwgPB
https://www.raymondcamden.com/2018/02/08/building-table-sorting-and-pagination-in-vuejs

*/

/* EventBus is used to pass changes between components */
// const EventBus = new Vue();

Vue.filter("decimalPlaces", (value, num = 2) => {
    if (value == null) {
        return "N/A";
    } else {
        return value.toFixed(num);
    }
});

Vue.component('abc-table', {
    data: function () {
        if (typeof window.sequence === 'undefined') {
            window.sequence = this.$root.sequence;
        }
        return {
            abc_models: this.$root.abc_models,
            // Column definitions in display order. Columns with no data across
            // any model are hidden (see visibleColumns). type: 'link' renders a
            // link, 'decimal' formats a number, 'scores' is an expandable list
            // of per-interface values.
            columns: [
                { key: 'model_id', label: 'Model Name', type: 'link',
                  title: 'The name of the model' },
                { key: 'model_source', label: 'Model Source',
                  title: 'The source of the model' },
                { key: 'avg_plddt', label: 'Average pLDDT', type: 'decimal',
                  decimals: 2,
                  title: 'The average pLDDT score of the model' },
                { key: 'ptm_score', label: 'pTM score',
                  title: 'The pTM score of the model' },
                { key: 'iptm_score', label: 'ipTM score',
                  title: 'The ipTM score of the model' },
                { key: 'ipsae_score', label: 'ipSAE', type: 'scores',
                  title: 'The best ipSAE (interaction prediction Score from '
                       + 'Aligned Errors) d0res_asym score for each interface' },
                { key: 'reactifptm_score', label: 'reactifPTM', type: 'scores',
                  title: 'The best reactifPTM (interface pTM over residues in '
                       + 'contact) for each interface' },
                { key: 'affinity_pred_value',
                  label: 'Affinity (log10 IC50 µM)', type: 'decimal',
                  decimals: 3,
                  title: 'Boltz-2 predicted binding affinity as log10 IC50 in '
                       + 'µM (lower = stronger binder)' },
                { key: 'affinity_probability_binary', label: 'Binder prob.',
                  type: 'decimal', decimals: 3,
                  title: 'Boltz-2 predicted probability the ligand is a '
                       + 'binder (0-1)' },
                { key: 'residue_clashes', label: 'Residue Clashes',
                  title: 'The number of possible residue clashes found in the '
                       + 'model - lower is better' },
                { key: 'atom_clashes', label: 'Atom Clashes',
                  title: 'The number of possible atom clashes found in the '
                       + 'model - lower is better' },
            ],
        }
    },
    computed: {
        // Only show columns where at least one model has a value, so runs
        // without a complex (no ipTM/ipSAE/reactifPTM) or without a ligand
        // (no affinity) don't render empty columns.
        visibleColumns() {
            const models = Object.values(this.abc_models || {});
            return this.columns.filter((col) =>
                models.some((m) => {
                    const v = m[col.key];
                    return v !== null && v !== undefined && v !== '';
                })
            );
        }
    },
    mounted() {
        let show = Object.keys(this.$root.abc_models).length !== 0;
        toggleDisplay("abc-title", show);
        toggleDisplay("div1", show);


        if (show) {
            this.$nextTick(() => {
                // Ensure sequence, ft1, and ABC_rowFeatureMap are defined and accessible
                if (typeof sequence !== 'undefined' && typeof ft1 !== 'undefined' && typeof ABC_rowFeatureMap !== 'undefined') {
                    sortTableAndFeatures('abc_table', sequence, ft1, ABC_rowFeatureMap, '#div1', 2);
                } else {
                    console.error("Required variables (sequence, ft1, ABC_rowFeatureMap) are not defined.");
                }
            });
        }
      },
      methods: {
        // Split a comma-separated score string ("AB:0.9,AC:0.8") into items.
        scoreItems(val) {
            if (val === null || val === undefined || val === '') return [];
            return String(val).split(',').filter((s) => s.length);
        },
        // The highest-scoring item, shown as the collapsed summary.
        bestScore(val) {
            const items = this.scoreItems(val);
            let best = '';
            let bestv = -Infinity;
            for (const it of items) {
                const p = it.split(':');
                const v = parseFloat(p[p.length - 1]);
                if (!isNaN(v) && v > bestv) { bestv = v; best = it; }
            }
            return best || (items[0] || '');
        },
        fmtDecimal(val, decimals) {
            if (val === null || val === undefined || val === '') return '';
            const n = Number(val);
            return isNaN(n) ? val : n.toFixed(decimals || 2);
        },
        sortCol(idx) {
            if (typeof sortTableAndFeatures === 'function') {
                sortTableAndFeatures('abc_table', sequence, ft1,
                                     ABC_rowFeatureMap, '#div1', idx);
            }
        },
        getButtonClass(modelSource) {
            switch (modelSource) {
                case 'AlphaFold3':
                    return 'btn-source1';
                case 'Boltz':
                    return 'btn-source2';
                case 'Chai-1':
                    return 'btn-source3';
                case 'Protenix':
                    return 'btn-source4';
                case 'OpenFold3':
                    return 'btn-source5';
                case 'RosettaFold3':
                    return 'btn-source6';
                default:
                    return 'btn-default';
            }
        }
    },
    template: `
    <div id="abc-table-container">
        <table id="abc_table">
            <thead>
                <tr>
                    <th v-for="(col, idx) in visibleColumns" :key="col.key"
                        :title="col.title" @click="sortCol(idx)">{{ col.label }}</th>
                    <th title="Link to a visualisation of the model and its corresponding PAE plot">Model visualisations</th>
                </tr>
            </thead>
            <tbody>
                <tr v-for="abcmodel in abc_models" :data-feature-name="abcmodel.model_id">
                    <td v-for="col in visibleColumns" :key="col.key">
                        <template v-if="col.type === 'link'">
                            <a v-bind:href="abcmodel.model_path" target="_blank">{{ abcmodel[col.key] }}</a>
                        </template>
                        <template v-else-if="col.type === 'decimal'">
                            {{ fmtDecimal(abcmodel[col.key], col.decimals) }}
                        </template>
                        <template v-else-if="col.type === 'scores' && scoreItems(abcmodel[col.key]).length > 3">
                            <details class="score-cell">
                                <summary>{{ bestScore(abcmodel[col.key]) }} <span class="score-more">(+{{ scoreItems(abcmodel[col.key]).length - 1 }} more)</span></summary>
                                <div class="score-list">
                                    <div v-for="item in scoreItems(abcmodel[col.key])" :key="item">{{ item }}</div>
                                </div>
                            </details>
                        </template>
                        <template v-else>
                            {{ abcmodel[col.key] }}
                        </template>
                    </td>
                    <td><a v-bind:href="abcmodel.pae_path" target="_blank"><button :class="getButtonClass(abcmodel.model_source)">Click for PAE Plot</button></a></td>
                </tr>
            </tbody>
        </table>
    </div>
    `
});

Vue.component('abc-feature-viewer', {
    data: function () {
        if (typeof window.sequence === 'undefined') {
            window.sequence = this.$root.sequence;
        }
        return {
            abc_models: this.$root.abc_models,
            chain_data: this.$root.chain_data,
            abc_features: [],
        }
    },
    template: `
      <div id="abc-feature-viewer-container" class="content"></div>
    `,
    methods: {
        addFeature(feature) {
            this.abc_features.push(feature);
            ft1.addFeature(feature);
            var rowId = feature.filter;
            window.ABC_rowFeatureMap[rowId] = feature;
        },
        generateABCFeatures() {
            const colors = {
                'v_low': '#FF7D45',
                'low': '#FFDB13',
                'confident': '#65CBF3',
                'v_high': '#0053D6'
            };

            const descriptions = {
                'v_low': 'Very Low Confidence (pLDDT < 50)',
                'low': 'Low Confidence (70 > pLDDT > 50)',
                'confident': 'Confident (90 > pLDDT > 70)',
                'v_high': 'Very High Confidence (pLDDT > 90)'
            };

            const chain_colours = [
                '#991999', // PyMol deeppurple (0.6, 0.1, 0.6)
                '#00BFBF', // PyMol teal (0, 0.75, 0.75)
                '#e9967a', // salmon
                '#009e73',
                '#f0e442',
                '#0072b2',
                '#d55e00',
                '#cc79a7'
            ];

            if (this.chain_data) {
                chain_data = [];
                let colorIndex = 0;
                const colorsLength = chain_colours.length;

                for (const [chain, [start, end]] of Object.entries(this.chain_data)) {
                    chain_data.push({
                        x: start,
                        y: end,
                        color: chain_colours[colorIndex],
                        description: chain
                    });
                    colorIndex = (colorIndex + 1) % colorsLength;
                }
            }

            let chain_feature = {
                data: chain_data,
                name: 'Chain Information',
                className: 'chains',
                type: "multipleRect",
                filter: 'chains',
            };
            this.addFeature(chain_feature);

            if (this.abc_models) {
                this.abc_models.forEach(model => {
                    const modelName = model.model_id;

                    let data = [];
                    for (const [confidence, regions] of Object.entries(model.plddt_regions)) {
                        regions.forEach(([start, end]) => {
                            data.push({
                                x: start,
                                y: end,
                                color: colors[confidence],
                                description: descriptions[confidence],
                            });
                        });
                    }
                    let feature = {
                        data: data,
                        name: modelName,
                        className: modelName,
                        type: "multipleRect",
                        filter: modelName,
                    };
                    this.addFeature(feature);
                });
            }
        }

    },
    mounted() {
        window.ABC_rowFeatureMap = {};
        var options = {
            showAxis: true,
            showSequence: true,
            brushActive: true,
            toolbar:true,
            bubbleHelp:true,
            zoomMax:10,
        };
        window.ft1 = new FeatureViewer.createFeature(sequence,"#div1", options);
        this.generateABCFeatures();

    selectTableAndFeatures(ft1);
    collapseDiv("collapsible1");
    }
});

new Vue({
    el: '#app',
    data: {
        abc_models: abc_data.models,
        sequence: abc_data.sequence,
        chain_data: abc_data.chain_data,
        plotly_path: abc_data.plotly_path,
    },
})
