const elements = [
  { atomicNumber: 0, symbol: "", name: "", category: "" },
  { atomicNumber: 1, symbol: "H", name: "Hydrogen", category: "Nonmetal" },
  { atomicNumber: 2, symbol: "He", name: "Helium", category: "Noble Gas" },
  { atomicNumber: 3, symbol: "Li", name: "Lithium", category: "Alkali Metal" },
  {
    atomicNumber: 4,
    symbol: "Be",
    name: "Beryllium",
    category: "Alkaline Earth Metal",
  },
  { atomicNumber: 5, symbol: "B", name: "Boron", category: "Metalloid" },
  { atomicNumber: 6, symbol: "C", name: "Carbon", category: "Nonmetal" },
  { atomicNumber: 7, symbol: "N", name: "Nitrogen", category: "Nonmetal" },
  { atomicNumber: 8, symbol: "O", name: "Oxygen", category: "Nonmetal" },
  { atomicNumber: 9, symbol: "F", name: "Fluorine", category: "Halogen" },
  { atomicNumber: 10, symbol: "Ne", name: "Neon", category: "Noble Gas" },
  { atomicNumber: 11, symbol: "Na", name: "Sodium", category: "Alkali Metal" },
  {
    atomicNumber: 12,
    symbol: "Mg",
    name: "Magnesium",
    category: "Alkaline Earth Metal",
  },
  {
    atomicNumber: 13,
    symbol: "Al",
    name: "Aluminium",
    category: "Post-transition Metal",
  },
  { atomicNumber: 14, symbol: "Si", name: "Silicon", category: "Metalloid" },
  { atomicNumber: 15, symbol: "P", name: "Phosphorus", category: "Nonmetal" },
  { atomicNumber: 16, symbol: "S", name: "Sulfur", category: "Nonmetal" },
  { atomicNumber: 17, symbol: "Cl", name: "Chlorine", category: "Halogen" },
  { atomicNumber: 18, symbol: "Ar", name: "Argon", category: "Noble Gas" },
  {
    atomicNumber: 19,
    symbol: "K",
    name: "Potassium",
    category: "Alkali Metal",
  },
  {
    atomicNumber: 20,
    symbol: "Ca",
    name: "Calcium",
    category: "Alkaline Earth Metal",
  },
  {
    atomicNumber: 21,
    symbol: "Sc",
    name: "Scandium",
    category: "Transition Metal",
  },
  {
    atomicNumber: 22,
    symbol: "Ti",
    name: "Titanium",
    category: "Transition Metal",
  },
  {
    atomicNumber: 23,
    symbol: "V",
    name: "Vanadium",
    category: "Transition Metal",
  },
  {
    atomicNumber: 24,
    symbol: "Cr",
    name: "Chromium",
    category: "Transition Metal",
  },
  {
    atomicNumber: 25,
    symbol: "Mn",
    name: "Manganese",
    category: "Transition Metal",
  },
  {
    atomicNumber: 26,
    symbol: "Fe",
    name: "Iron",
    category: "Transition Metal",
  },
  {
    atomicNumber: 27,
    symbol: "Co",
    name: "Cobalt",
    category: "Transition Metal",
  },
  {
    atomicNumber: 28,
    symbol: "Ni",
    name: "Nickel",
    category: "Transition Metal",
  },
  {
    atomicNumber: 29,
    symbol: "Cu",
    name: "Copper",
    category: "Transition Metal",
  },
  {
    atomicNumber: 30,
    symbol: "Zn",
    name: "Zinc",
    category: "Transition Metal",
  },
  {
    atomicNumber: 31,
    symbol: "Ga",
    name: "Gallium",
    category: "Post-transition Metal",
  },
  { atomicNumber: 32, symbol: "Ge", name: "Germanium", category: "Metalloid" },
  { atomicNumber: 33, symbol: "As", name: "Arsenic", category: "Metalloid" },
  { atomicNumber: 34, symbol: "Se", name: "Selenium", category: "Nonmetal" },
  { atomicNumber: 35, symbol: "Br", name: "Bromine", category: "Halogen" },
  { atomicNumber: 36, symbol: "Kr", name: "Krypton", category: "Noble Gas" },
  {
    atomicNumber: 37,
    symbol: "Rb",
    name: "Rubidium",
    category: "Alkali Metal",
  },
  {
    atomicNumber: 38,
    symbol: "Sr",
    name: "Strontium",
    category: "Alkaline Earth Metal",
  },
  {
    atomicNumber: 39,
    symbol: "Y",
    name: "Yttrium",
    category: "Transition Metal",
  },
  {
    atomicNumber: 40,
    symbol: "Zr",
    name: "Zirconium",
    category: "Transition Metal",
  },
  {
    atomicNumber: 41,
    symbol: "Nb",
    name: "Niobium",
    category: "Transition Metal",
  },
  {
    atomicNumber: 42,
    symbol: "Mo",
    name: "Molybdenum",
    category: "Transition Metal",
  },
  {
    atomicNumber: 43,
    symbol: "Tc",
    name: "Technetium",
    category: "Transition Metal",
  },
  {
    atomicNumber: 44,
    symbol: "Ru",
    name: "Ruthenium",
    category: "Transition Metal",
  },
  {
    atomicNumber: 45,
    symbol: "Rh",
    name: "Rhodium",
    category: "Transition Metal",
  },
  {
    atomicNumber: 46,
    symbol: "Pd",
    name: "Palladium",
    category: "Transition Metal",
  },
  {
    atomicNumber: 47,
    symbol: "Ag",
    name: "Silver",
    category: "Transition Metal",
  },
  {
    atomicNumber: 48,
    symbol: "Cd",
    name: "Cadmium",
    category: "Transition Metal",
  },
  {
    atomicNumber: 49,
    symbol: "In",
    name: "Indium",
    category: "Post-transition Metal",
  },
  {
    atomicNumber: 50,
    symbol: "Sn",
    name: "Tin",
    category: "Post-transition Metal",
  },
  { atomicNumber: 51, symbol: "Sb", name: "Antimony", category: "Metalloid" },
  { atomicNumber: 52, symbol: "Te", name: "Tellurium", category: "Metalloid" },
  { atomicNumber: 53, symbol: "I", name: "Iodine", category: "Halogen" },
  { atomicNumber: 54, symbol: "Xe", name: "Xenon", category: "Noble Gas" },
  { atomicNumber: 55, symbol: "Cs", name: "Caesium", category: "Alkali Metal" },
  {
    atomicNumber: 56,
    symbol: "Ba",
    name: "Barium",
    category: "Alkaline Earth Metal",
  },
  // ... Lanthanides ...
  { atomicNumber: 57, symbol: "La", name: "Lanthanum", category: "Lanthanide" },
  { atomicNumber: 58, symbol: "Ce", name: "Cerium", category: "Lanthanide" },
  {
    atomicNumber: 59,
    symbol: "Pr",
    name: "Praseodymium",
    category: "Lanthanide",
  },
  { atomicNumber: 60, symbol: "Nd", name: "Neodymium", category: "Lanthanide" },
  {
    atomicNumber: 61,
    symbol: "Pm",
    name: "Promethium",
    category: "Lanthanide",
  },
  { atomicNumber: 62, symbol: "Sm", name: "Samarium", category: "Lanthanide" },
  { atomicNumber: 63, symbol: "Eu", name: "Europium", category: "Lanthanide" },
  {
    atomicNumber: 64,
    symbol: "Gd",
    name: "Gadolinium",
    category: "Lanthanide",
  },
  { atomicNumber: 65, symbol: "Tb", name: "Terbium", category: "Lanthanide" },
  {
    atomicNumber: 66,
    symbol: "Dy",
    name: "Dysprosium",
    category: "Lanthanide",
  },
  { atomicNumber: 67, symbol: "Ho", name: "Holmium", category: "Lanthanide" },
  { atomicNumber: 68, symbol: "Er", name: "Erbium", category: "Lanthanide" },
  { atomicNumber: 69, symbol: "Tm", name: "Thulium", category: "Lanthanide" },
  { atomicNumber: 70, symbol: "Yb", name: "Ytterbium", category: "Lanthanide" },
  { atomicNumber: 71, symbol: "Lu", name: "Lutetium", category: "Lanthanide" },
  {
    atomicNumber: 72,
    symbol: "Hf",
    name: "Hafnium",
    category: "Transition Metal",
  },
  {
    atomicNumber: 73,
    symbol: "Ta",
    name: "Tantalum",
    category: "Transition Metal",
  },
  {
    atomicNumber: 74,
    symbol: "W",
    name: "Tungsten",
    category: "Transition Metal",
  },
  {
    atomicNumber: 75,
    symbol: "Re",
    name: "Rhenium",
    category: "Transition Metal",
  },
  {
    atomicNumber: 76,
    symbol: "Os",
    name: "Osmium",
    category: "Transition Metal",
  },
  {
    atomicNumber: 77,
    symbol: "Ir",
    name: "Iridium",
    category: "Transition Metal",
  },
  {
    atomicNumber: 78,
    symbol: "Pt",
    name: "Platinum",
    category: "Transition Metal",
  },
  {
    atomicNumber: 79,
    symbol: "Au",
    name: "Gold",
    category: "Transition Metal",
  },
  {
    atomicNumber: 80,
    symbol: "Hg",
    name: "Mercury",
    category: "Transition Metal",
  },
  {
    atomicNumber: 81,
    symbol: "Tl",
    name: "Thallium",
    category: "Post-transition Metal",
  },
  {
    atomicNumber: 82,
    symbol: "Pb",
    name: "Lead",
    category: "Post-transition Metal",
  },
  {
    atomicNumber: 83,
    symbol: "Bi",
    name: "Bismuth",
    category: "Post-transition Metal",
  },
  { atomicNumber: 84, symbol: "Po", name: "Polonium", category: "Metalloid" },
  { atomicNumber: 85, symbol: "At", name: "Astatine", category: "Halogen" },
  { atomicNumber: 86, symbol: "Rn", name: "Radon", category: "Noble Gas" },
  {
    atomicNumber: 87,
    symbol: "Fr",
    name: "Francium",
    category: "Alkali Metal",
  },
  {
    atomicNumber: 88,
    symbol: "Ra",
    name: "Radium",
    category: "Alkaline Earth Metal",
  },
  // ... Actinides ...
  { atomicNumber: 89, symbol: "Ac", name: "Actinium", category: "Actinide" },
  { atomicNumber: 90, symbol: "Th", name: "Thorium", category: "Actinide" },
  {
    atomicNumber: 91,
    symbol: "Pa",
    name: "Protactinium",
    category: "Actinide",
  },
  { atomicNumber: 92, symbol: "U", name: "Uranium", category: "Actinide" },
  { atomicNumber: 93, symbol: "Np", name: "Neptunium", category: "Actinide" },
  { atomicNumber: 94, symbol: "Pu", name: "Plutonium", category: "Actinide" },
  { atomicNumber: 95, symbol: "Am", name: "Americium", category: "Actinide" },
  { atomicNumber: 96, symbol: "Cm", name: "Curium", category: "Actinide" },
  { atomicNumber: 97, symbol: "Bk", name: "Berkelium", category: "Actinide" },
  { atomicNumber: 98, symbol: "Cf", name: "Californium", category: "Actinide" },
  { atomicNumber: 99, symbol: "Es", name: "Einsteinium", category: "Actinide" },
  { atomicNumber: 100, symbol: "Fm", name: "Fermium", category: "Actinide" },
  {
    atomicNumber: 101,
    symbol: "Md",
    name: "Mendelevium",
    category: "Actinide",
  },
  { atomicNumber: 102, symbol: "No", name: "Nobelium", category: "Actinide" },
  { atomicNumber: 103, symbol: "Lr", name: "Lawrencium", category: "Actinide" },
  {
    atomicNumber: 104,
    symbol: "Rf",
    name: "Rutherfordium",
    category: "Transition Metal",
  },
  {
    atomicNumber: 105,
    symbol: "Db",
    name: "Dubnium",
    category: "Transition Metal",
  },
  {
    atomicNumber: 106,
    symbol: "Sg",
    name: "Seaborgium",
    category: "Transition Metal",
  },
  {
    atomicNumber: 107,
    symbol: "Bh",
    name: "Bohrium",
    category: "Transition Metal",
  },
  {
    atomicNumber: 108,
    symbol: "Hs",
    name: "Hassium",
    category: "Transition Metal",
  },
  {
    atomicNumber: 109,
    symbol: "Mt",
    name: "Meitnerium",
    category: "Transition Metal",
  },
  {
    atomicNumber: 110,
    symbol: "Ds",
    name: "Darmstadtium",
    category: "Transition Metal",
  },
  {
    atomicNumber: 111,
    symbol: "Rg",
    name: "Roentgenium",
    category: "Transition Metal",
  },
  {
    atomicNumber: 112,
    symbol: "Cn",
    name: "Copernicium",
    category: "Transition Metal",
  },
  {
    atomicNumber: 113,
    symbol: "Nh",
    name: "Nihonium",
    category: "Post-transition Metal",
  },
  {
    atomicNumber: 114,
    symbol: "Fl",
    name: "Flerovium",
    category: "Post-transition Metal",
  },
  {
    atomicNumber: 115,
    symbol: "Mc",
    name: "Moscovium",
    category: "Post-transition Metal",
  },
  {
    atomicNumber: 116,
    symbol: "Lv",
    name: "Livermorium",
    category: "Post-transition Metal",
  },
  { atomicNumber: 117, symbol: "Ts", name: "Tennessine", category: "Halogen" },
  { atomicNumber: 118, symbol: "Og", name: "Oganesson", category: "Noble Gas" },
];

export const periodicTableRows = [
  [elements[1], ...Array(16).fill(elements[0]), elements[2]],
  [
    elements[3],
    elements[4],
    ...Array(10).fill(elements[0]),
    elements[5],
    elements[6],
    elements[7],
    elements[8],
    elements[9],
    elements[10],
  ],
  [
    elements[11],
    elements[12],
    ...Array(10).fill(elements[0]),
    elements[13],
    elements[14],
    elements[15],
    elements[16],
    elements[17],
    elements[18],
  ],
  [
    elements[19],
    elements[20],
    elements[21],
    elements[22],
    elements[23],
    elements[24],
    elements[25],
    elements[26],
    elements[27],
    elements[28],
    elements[29],
    elements[30],
    elements[31],
    elements[32],
    elements[33],
    elements[34],
    elements[35],
    elements[36],
  ],
  [
    elements[37],
    elements[38],
    elements[39],
    elements[40],
    elements[41],
    elements[42],
    elements[43],
    elements[44],
    elements[45],
    elements[46],
    elements[47],
    elements[48],
    elements[49],
    elements[50],
    elements[51],
    elements[52],
    elements[53],
    elements[54],
  ],
  [
    elements[55],
    elements[56],
    elements[57],
    elements[72],
    elements[73],
    elements[74],
    elements[75],
    elements[76],
    elements[77],
    elements[78],
    elements[79],
    elements[80],
    elements[81],
    elements[82],
    elements[83],
    elements[84],
    elements[85],
    elements[86],
  ],
  [
    elements[87],
    elements[88],
    elements[89],
    elements[104],
    elements[105],
    elements[106],
    elements[107],
    elements[108],
    elements[109],
    elements[110],
    elements[111],
    elements[112],
    elements[113],
    elements[114],
    elements[115],
    elements[116],
    elements[117],
    elements[118],
  ],
  // Lanthanides (separate row)
  elements.slice(57, 72),
  // Actinides (separate row)
  elements.slice(89, 104),
];