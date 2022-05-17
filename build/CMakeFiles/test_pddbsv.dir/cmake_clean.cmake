file(REMOVE_RECURSE
  "../bin/test_pddbsv.pdb"
  "../bin/test_pddbsv"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/test_pddbsv.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
