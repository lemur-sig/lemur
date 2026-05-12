clean:
	cd lemur-py && $(MAKE) clean
	cd lemur-rs && cargo clean	
	cd parameter && $(MAKE) clean

