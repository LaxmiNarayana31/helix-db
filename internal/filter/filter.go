package filter

import "reflect"

type Filter map[string]interface{}

// Match returns true if all key-value pairs in the filter are exactly matched in the metadata.
func (f Filter) Match(metadata map[string]interface{}) bool {
	if len(f) == 0 {
		return true
	}
	if metadata == nil {
		return false
	}
	for k, v := range f {
		metaVal, ok := metadata[k]
		if !ok {
			return false
		}

		// Handle numeric coercions where metaVal might be float64 from JSON but v is an integer type from Go
		vFloat, ok1 := toFloat64(v)
		metaFloat, ok2 := toFloat64(metaVal)
		if ok1 && ok2 {
			if vFloat != metaFloat {
				return false
			}
			continue
		}

		if !reflect.DeepEqual(metaVal, v) {
			return false
		}
	}
	return true
}

func toFloat64(val interface{}) (float64, bool) {
	switch v := val.(type) {
	case int:
		return float64(v), true
	case int8:
		return float64(v), true
	case int16:
		return float64(v), true
	case int32:
		return float64(v), true
	case int64:
		return float64(v), true
	case uint:
		return float64(v), true
	case uint8:
		return float64(v), true
	case uint16:
		return float64(v), true
	case uint32:
		return float64(v), true
	case uint64:
		return float64(v), true
	case float32:
		return float64(v), true
	case float64:
		return v, true
	default:
		return 0, false
	}
}
