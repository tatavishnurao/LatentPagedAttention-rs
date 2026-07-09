#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheConfig {
    pub block_size: usize,
    pub num_layers: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
}

impl CacheConfig {
    pub fn new(
        block_size: usize,
        num_layers: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self, &'static str> {
        if block_size == 0 || num_layers == 0 || kv_heads == 0 || head_dim == 0 {
            return Err("cache config values must be > 0");
        }
        Ok(Self {
            block_size,
            num_layers,
            kv_heads,
            head_dim,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockLocation {
    pub physical_block: usize,
    pub offset: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockTable {
    block_size: usize,
    logical_to_physical: Vec<usize>,
    next_physical_block: usize,
    num_tokens: usize,
}

impl BlockTable {
    pub fn new(block_size: usize) -> Result<Self, &'static str> {
        if block_size == 0 {
            return Err("block_size must be > 0");
        }
        Ok(Self {
            block_size,
            logical_to_physical: Vec::new(),
            next_physical_block: 0,
            num_tokens: 0,
        })
    }

    pub fn allocate_tokens(
        &mut self,
        count: usize,
    ) -> Result<Vec<(usize, BlockLocation)>, &'static str> {
        if count == 0 {
            return Err("count must be > 0");
        }

        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            let token_pos = self.num_tokens;
            let logical_block = token_pos / self.block_size;
            if logical_block == self.logical_to_physical.len() {
                self.logical_to_physical.push(self.next_physical_block);
                self.next_physical_block += 1;
            }
            self.num_tokens += 1;
            let location = self.translate(token_pos)?;
            out.push((token_pos, location));
        }
        Ok(out)
    }

    pub fn translate(&self, token_pos: usize) -> Result<BlockLocation, &'static str> {
        if token_pos >= self.num_tokens {
            return Err("token position is not allocated");
        }
        let logical_block = token_pos / self.block_size;
        Ok(BlockLocation {
            physical_block: self.logical_to_physical[logical_block],
            offset: token_pos % self.block_size,
        })
    }

    pub fn num_blocks(&self) -> usize {
        self.logical_to_physical.len()
    }
}

#[cfg(test)]
mod tests {
    use super::{BlockTable, CacheConfig};

    #[test]
    fn cache_config_rejects_zero_values() {
        assert!(CacheConfig::new(0, 28, 8, 128).is_err());
    }

    #[test]
    fn block_table_allocates_across_boundaries() {
        let mut table = BlockTable::new(4).unwrap();
        let allocations = table.allocate_tokens(6).unwrap();

        assert_eq!(allocations[0].0, 0);
        assert_eq!(allocations[0].1.physical_block, 0);
        assert_eq!(allocations[3].1.offset, 3);
        assert_eq!(allocations[4].1.physical_block, 1);
        assert_eq!(allocations[4].1.offset, 0);
        assert_eq!(table.num_blocks(), 2);
        assert_eq!(table.translate(5).unwrap().offset, 1);
    }

    #[test]
    fn block_table_rejects_out_of_range_token() {
        let mut table = BlockTable::new(8).unwrap();
        table.allocate_tokens(2).unwrap();
        assert!(table.translate(2).is_err());
    }
}
