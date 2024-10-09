// INTERFACE API FOR LLM-BASED BABA AGENTS

function loadJsonFromFile(filePath) {
    try {
        // Read the file synchronously
        const rawData = fs.readFileSync(filePath, 'utf8');

        // Parse the JSON data
        const jsonData = JSON.parse(rawData);

        return jsonData;
    } catch (error) {
        console.error('Error reading or parsing file:', error);
        return null;
    }
}

// Modified from ./js/json_io.js
function saveJsonToFile(jsonData, filePath) {
    // Convert JSON object to string
    const data = JSON.stringify(jsonData, null, 2);

    // Write JSON string to a file
    fs.writeFile(filePath, data, (err) => {
      if (err) throw err;
    });
}

function importLevels(lvlsetJSON, levelsDir){
    let path = levelsDir + '/' + lvlsetJSON + ".json";
    if (!fs.existsSync(path)){return null;}     //no level set found

    let j = fs.readFileSync(path);
    let lvlset = JSON.parse(j);
    return lvlset.levels;
}

//file I/O
const fs = require('fs');

//import simulation code
const simjs = require('./js/simulation');

/**
 * Process an action and update the game state.
 * @param {string} action - The action to be performed (e.g., 'up', 'down', 'left', 'right', 'space').
 * @return {Object} The updated game state and any relevant information (e.g., whether the level is won).
 */
function processAction(action, gameState) {
    let actionResult = simjs.nextMove(action, gameState);
    newGameState = actionResult.next_state;

    // Return the updated state and whether the level is won
    return {
        state: newGameState,
        won: actionResult.won
    };
}

let args = process.argv;
let statePath = args[2];
let resultPath = args[3];
let levelSet = args[4]; // e.g., demo_LEVELS
let levelID = parseInt(args[5]);
let move = args[6];

let levelsDir = 'KekeCompetition-main/Keke_JS/json_levels'

let gameState

//setup state
// simjs.setupLevel(simjs.parseMap(ascii_map));
// game_state = simjs.getGamestate();

if (statePath === 'None') {
    // Load levels
    const levels = importLevels(levelSet, levelsDir);
    // Convert ASCII map string to 2D array format
    let initialMap = simjs.parseMap(levels[levelID].ascii);

    simjs.setupLevel(initialMap)
    // Initialize game state
    gameState = simjs.getGamestate();
} else {
    let state_ascii = loadJsonFromFile(statePath).asciiMap;

    // Convert ASCII map string to 2D array format
    let initialMap = simjs.parseMap(state_ascii);

    simjs.setupLevel(initialMap)
    // Initialize game state
    gameState = simjs.getGamestate();


}

if (move !== 'None') {
    let result = processAction(move, gameState);
    // result = simjs.doubleMap2Str(result)
    // doubleMap2Str
    // Convert obj_map and back_map to ASCII representation
    let asciiMap = simjs.doubleMap2Str(result.state.obj_map, result.state.back_map);
    
    // Save the ASCII map to a file
    fs.writeFileSync('output_ascii_map.txt', asciiMap);

    // Save the minimal initial state
    let result_full = {
        ascii_map: asciiMap,
        result: result,
        won: result.won
    };

    saveJsonToFile(result_full, resultPath)

    // console.log(result.won)
} else {

    let result = processAction("space", gameState);

    let asciiMap = simjs.doubleMap2Str(result.state.obj_map, result.state.back_map);

    // Save the minimal initial state
    let result_full = {
        ascii_map: asciiMap,
        result: result,
        won: false
    };

    saveJsonToFile(result_full, resultPath)
}
