// Import necessary modules
const fs = require('fs');
const simjs = require('./js/simulation');

// Function to load JSON from a file
function loadJsonFromFile(filePath) {
    try {
        const rawData = fs.readFileSync(filePath, 'utf8');
        return JSON.parse(rawData);
    } catch (error) {
        console.error('Error reading or parsing file:', error);
        return null;
    }
}

// Function to save JSON to a file
function saveJsonToFile(jsonData, filePath) {
    const data = JSON.stringify(jsonData, null, 2);
    fs.writeFile(filePath, data, (err) => {
        if (err) throw err;
    });
}

// Function to import levels
function importLevels(lvlsetJSON, levelsDir) {
    const path = `${levelsDir}/${lvlsetJSON}.json`;
    if (!fs.existsSync(path)) {
        console.error('Level set not found:', path);
        return null;
    }
    const j = fs.readFileSync(path);
    const lvlset = JSON.parse(j);
    return lvlset.levels;
}

// Initialize the map using ASCII representation
function initMap(ascii_map) {
    simjs.setupLevel(simjs.parseMap(ascii_map));
    return simjs.doubleMap2Str(simjs.getGamestate().obj_map, simjs.getGamestate().back_map);
}

// Update the ASCII map by processing the action
function updateMap(ascii_map, action) {
    simjs.setupLevel(simjs.parseMap(ascii_map));
    const gameState = simjs.getGamestate();
    const res = simjs.nextMove(action, gameState);

    const updatedAsciiMap = simjs.doubleMap2Str(res.next_state.obj_map, res.next_state.back_map);

    // Return the updated ASCII map and whether the game is won
    return {
        ascii_map: updatedAsciiMap,
        state: res,
        won: res.won
    };
}

let args = process.argv;
let statePath = args[2];
let resultPath = args[3];
let levelSet = args[4]; // e.g., demo_LEVELS
let levelID = parseInt(args[5]);
let move = args[6];

let levelsDir = 'KekeCompetition-main/Keke_JS/json_levels';
let ascii_map;

if (statePath === 'None') {
    // Load levels
    const levels = importLevels(levelSet, levelsDir);
    ascii_map = levels[levelID].ascii;

    // Initialize and get the ASCII map
    ascii_map = initMap(ascii_map);
} else {
    // Load the ASCII map from the provided state file
    ascii_map = loadJsonFromFile(statePath).ascii_map;
}

if (move !== 'None') {
    let result = updateMap(ascii_map, move);
    
    // Save the updated ASCII map
    fs.writeFileSync('output_ascii_map.txt', result.ascii_map);
    saveJsonToFile(result, resultPath);
} else {
    // If no move, just save the current ASCII map
    let result = updateMap(ascii_map, "space");

    // return {
    //     ascii_map: updatedAsciiMap,
    //     state: result,
    //     won: false
    // };

    // saveJsonToFile({ ascii_map: ascii_map }, resultPath);
    saveJsonToFile(result, resultPath);

}
