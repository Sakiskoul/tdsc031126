pragma solidity 0.8.24;


contract AlarmOracle {
    uint256 public constant LEN1 = 64;
    uint256 public constant LEN2 = 3;
    uint256 public constant LEN = LEN1 + LEN2;
    uint256 public constant W1 = 15;
    uint256 public constant H = 10;

    bytes32[] public roots;
    uint256 public immutable k;
    uint256 public immutable startWeek;

    uint8 public threatLevel;
    uint256 public levelWeek;
    mapping(uint256 => mapping(uint256 => bool)) public used;

    event Attested(uint256 indexed week, uint8 level, bytes32 digest, uint256 signers);

    struct OpSig {
        uint256 op;
        bytes32[67] sig;
        bytes32[10] path;
    }

    constructor(bytes32[] memory _roots, uint256 _k, uint256 _startWeek) {
        require(_k > 0 && _k <= _roots.length, "bad quorum");
        roots = _roots;
        k = _k;
        startWeek = _startWeek;
    }

    function statement(uint256 week, uint8 level, bytes32 digest) public view returns (bytes32) {
        return keccak256(abi.encodePacked("ARGUS-ALARM-v1", block.chainid, address(this), week, level, digest));
    }


    function _f(bytes32 x, uint256 op, uint256 week, uint256 i, uint256 j) private pure returns (bytes32) {
        return keccak256(abi.encodePacked(x, uint64(op), uint64(week), uint16(i), uint8(j)));
    }

    function _digits(bytes32 m) private pure returns (uint8[67] memory d) {
        uint256 csum = 0;
        for (uint256 i = 0; i < LEN1; i++) {
            uint8 b = uint8(m[i / 2]);
            uint8 v = (i % 2 == 0) ? (b >> 4) : (b & 0x0f);
            d[i] = v;
            csum += W1 - v;
        }
        d[64] = uint8((csum >> 8) & 0x0f);
        d[65] = uint8((csum >> 4) & 0x0f);
        d[66] = uint8(csum & 0x0f);
    }


    function wotsPk(bytes32 m, bytes32[67] calldata sig, uint256 op, uint256 week) public pure returns (bytes32) {
        uint8[67] memory d = _digits(m);
        bytes32[67] memory ends;
        for (uint256 i = 0; i < LEN; i++) {
            bytes32 x = sig[i];
            for (uint256 j = d[i]; j < W1; j++) {
                x = _f(x, op, week, i, j);
            }
            ends[i] = x;
        }
        return keccak256(abi.encodePacked(ends));
    }

    function _root(bytes32 leaf, uint256 index, bytes32[10] calldata path) private pure returns (bytes32 h) {
        h = leaf;
        for (uint256 l = 0; l < H; l++) {
            h = ((index >> l) & 1) == 0
                ? keccak256(abi.encodePacked(h, path[l]))
                : keccak256(abi.encodePacked(path[l], h));
        }
    }

    function submitAttestation(uint256 week, uint8 level, bytes32 digest, OpSig[] calldata sigs) external {
        require(week >= startWeek && week < startWeek + (1 << H), "week out of range");
        require(week >= levelWeek, "stale week");
        bytes32 m = statement(week, level, digest);
        uint256 idx = week - startWeek;
        uint256 seen = 0;
        uint256 valid = 0;
        for (uint256 s = 0; s < sigs.length; s++) {
            uint256 op = sigs[s].op;
            require(op < roots.length && op < 256, "bad operator");
            require((seen >> op) & 1 == 0, "duplicate operator");
            seen |= (1 << op);
            require(!used[op][week], "key already used");
            bytes32 leaf = keccak256(abi.encodePacked(wotsPk(m, sigs[s].sig, op, week)));
            if (_root(leaf, idx, sigs[s].path) == roots[op]) {
                used[op][week] = true;
                valid++;
            }
        }
        require(valid >= k, "quorum not reached");
        threatLevel = level;
        levelWeek = week;
        emit Attested(week, level, digest, valid);
    }
}
